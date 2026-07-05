from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import featurize as F
from .cards import ATK_DIM, ATTR_DIM

VALUE_ROWS = (3, 4)


@dataclass
class ModelConfig:
    d: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 2048
    dec_layers: int = 2
    dropout: float = 0.0  # nonzero would break padding invariance and the ratio contract
    n_card_rows: int = 0
    n_attack_rows: int = 0


def teacher_config(tables):
    return ModelConfig(n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


def student_config(tables):
    return ModelConfig(d=224, layers=4, heads=8, ffn=896, dec_layers=1,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


def tiny_config(tables):
    return ModelConfig(d=64, layers=2, heads=4, ffn=128, dec_layers=1,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


class CardEmbed(nn.Module):
    """Shared card-identity encoding: id embedding + printed-attribute projection."""

    def __init__(self, cfg, attr_table: np.ndarray):
        super().__init__()
        self.emb = nn.Embedding(cfg.n_card_rows, cfg.d, padding_idx=0)
        self.register_buffer("attr", torch.from_numpy(attr_table))
        self.attr_proj = nn.Linear(ATTR_DIM, cfg.d)

    def forward(self, rows):
        return self.emb(rows) + self.attr_proj(self.attr[rows])


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None):
        super().__init__()
        if attr_table is None:
            from .cards import build_tables
            attr_table = build_tables().attr
        self.cfg = cfg
        self.card = CardEmbed(cfg, attr_table)
        self.num = nn.Linear(F.NUM_DIM, cfg.d)
        self.owner = nn.Embedding(F.N_OWNER, cfg.d)
        self.zone = nn.Embedding(F.N_ZONE, cfg.d)
        self.kind = nn.Embedding(F.N_KIND, cfg.d)
        self.pos = nn.Embedding(F.N_POS, cfg.d)
        self.norm = nn.LayerNorm(cfg.d)
        layer = nn.TransformerEncoderLayer(
            cfg.d, cfg.heads, cfg.ffn, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, cfg.layers)

    def forward(self, batch):
        x = (self.card(batch["card"]) + self.num(batch["numeric"])
             + self.owner(batch["owner"]) + self.zone(batch["zone"])
             + self.kind(batch["kind"]) + self.pos(batch["pos"]))
        x = self.norm(x)
        return self.trunk(x, src_key_padding_mask=~batch["mask"])


def collate_states(states):
    def stack(name, dtype):
        return torch.stack([torch.as_tensor(getattr(s, name)).to(dtype) for s in states])
    return {
        "card": stack("card", torch.int64),
        "numeric": stack("numeric", torch.float32),
        "owner": stack("owner", torch.int64),
        "zone": stack("zone", torch.int64),
        "kind": stack("kind", torch.int64),
        "pos": stack("pos", torch.int64),
        "mask": stack("mask", torch.bool),
    }


class PolicyModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None, attack_table=None):
        super().__init__()
        if attack_table is None:
            from .cards import build_tables
            t = build_tables()
            attr_table, attack_table = t.attr, t.attack_feat
        self.cfg = cfg
        self.encoder = Encoder(cfg, attr_table)
        self.register_buffer("atk", torch.from_numpy(attack_table))
        self.opt_type = nn.Embedding(F.N_OPT_TYPE, cfg.d)
        self.q_type = nn.Embedding(F.N_SELECT_TYPE, cfg.d)
        self.q_ctx = nn.Embedding(F.N_SELECT_CTX, cfg.d)
        self.atk_proj = nn.Linear(ATK_DIM, cfg.d)
        # attack identity (spec: attackId -> attack embedding); printed stats
        # alone would alias same-stat attacks. padding_idx=0 keeps non-attack
        # options (PAD row) at zero contribution.
        self.atk_emb = nn.Embedding(cfg.n_attack_rows, cfg.d, padding_idx=0)
        self.opt_scalar = nn.Linear(F.OPT_SCALAR_DIM, cfg.d)
        self.q_scalar = nn.Linear(F.Q_SCALAR_DIM, cfg.d)
        # role projection for an option's second (target) reference, so a
        # source ref and a target ref pointing at the same token don't alias
        self.ref2_proj = nn.Linear(cfg.d, cfg.d)
        self.picked_proj = nn.Linear(cfg.d, cfg.d)
        self.done_tok = nn.Parameter(torch.zeros(cfg.d))
        self.opt_norm = nn.LayerNorm(cfg.d)
        dec = nn.TransformerDecoderLayer(
            cfg.d, cfg.heads, cfg.ffn, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec, cfg.dec_layers)
        self.logit = nn.Linear(cfg.d, 1)
        self.v_head = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(),
                                    nn.Linear(cfg.d, 1), nn.Tanh())
        self.pd_head = nn.Linear(2 * cfg.d, 1)
        self.dl_head = nn.Linear(cfg.d, cfg.n_card_rows)
        self.hd_head = nn.Linear(cfg.d, cfg.n_card_rows)
        # Zero-init the value heads' final linear layers so initial value
        # predictions are exactly 0 and the first optimizer steps on the MSE
        # terms cannot overshoot (standard RL value-head stabilization).
        nn.init.zeros_(self.v_head[2].weight)
        nn.init.zeros_(self.v_head[2].bias)
        nn.init.zeros_(self.pd_head.weight)
        nn.init.zeros_(self.pd_head.bias)

    def encode(self, state_batch):
        return self.encoder(state_batch)

    def _gather_ref(self, trunk, ref):
        safe = ref.clamp(min=0)
        g = torch.gather(trunk, 1, safe.unsqueeze(-1).expand(-1, -1, trunk.shape[-1]))
        return g * (ref >= 0).unsqueeze(-1)

    def option_logits(self, trunk, state_batch, sel, picked):
        B, O = sel["opt_type"].shape
        card_vec = self.encoder.card(sel["opt_card"])
        # mask ref2 AFTER the linear so absent (-1) refs contribute exact
        # zeros (a bare Linear would leak its bias into every option)
        ref2 = (self.ref2_proj(self._gather_ref(trunk, sel["opt_ref2"]))
                * (sel["opt_ref2"] >= 0).unsqueeze(-1))
        opt = (self.opt_type(sel["opt_type"]) + card_vec
               + self.atk_proj(self.atk[sel["opt_attack"]])
               + self.atk_emb(sel["opt_attack"])
               + self.opt_scalar(sel["opt_scalar"])
               + self._gather_ref(trunk, sel["opt_ref"])
               + ref2)
        q = (self.q_type(sel["q_type"]) + self.q_ctx(sel["q_ctx"])
             + self.q_scalar(sel["q_scalar"])
             + self._gather_ref(trunk, sel["q_ref"]).sum(1))
        picked_sum = (opt * picked[:, :O].unsqueeze(-1)).sum(1)
        q = q + self.picked_proj(picked_sum)
        done = self.done_tok.expand(B, 1, -1)
        tgt = self.opt_norm(torch.cat([q.unsqueeze(1), opt, done], dim=1))
        # q (col 0) and done (last col) are never padding; only the O option
        # columns can be, when this row's real option count < batch-wide O.
        # Without this mask, decoder self-attention lets padded option slots
        # leak into real option/q/done logits whenever the batch mixes
        # option counts (never exercised until batched replay).
        # POLARITY: True in tgt_key_padding_mask means IGNORE as a key, so
        # the never-padding q/done columns must be False — True there would
        # sever query/picked conditioning (q, done dropped from self-attn).
        not_padded = torch.zeros((B, 1), dtype=torch.bool, device=tgt.device)
        tgt_key_padding_mask = torch.cat(
            [not_padded, ~sel["opt_mask"], not_padded], dim=1)
        h = self.decoder(tgt, trunk, memory_key_padding_mask=~state_batch["mask"],
                          tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.logit(h[:, 1:, :]).squeeze(-1)          # [B, O+1]
        neg = float("-inf")
        pad = ~sel["opt_mask"]
        logits[:, :O] = logits[:, :O].masked_fill(pad | picked[:, :O], neg)
        n_picked = picked[:, :O].sum(-1)
        done_illegal = n_picked < sel["min_count_t"]
        logits[:, O] = logits[:, O].masked_fill(done_illegal, neg)
        return logits

    def _pooled(self, trunk):
        return torch.cat([trunk[:, VALUE_ROWS[0]], trunk[:, VALUE_ROWS[1]]], dim=-1)

    def public_value(self, trunk):
        return self.v_head(self._pooled(trunk)).squeeze(-1)

    def prize_diff(self, trunk):
        return self.pd_head(self._pooled(trunk)).squeeze(-1)

    def aux_decklist(self, trunk):
        return nn.functional.softplus(self.dl_head(trunk[:, 0]))

    def aux_hand(self, trunk):
        return nn.functional.softplus(self.hd_head(trunk[:, 0]))


def collate_selects(selects, device=None):
    B = len(selects)
    O = max(len(s.opt_type) for s in selects)
    out = {
        "opt_type": torch.zeros(B, O, dtype=torch.int64),
        "opt_ref": torch.full((B, O), -1, dtype=torch.int64),
        "opt_ref2": torch.full((B, O), -1, dtype=torch.int64),
        "opt_card": torch.zeros(B, O, dtype=torch.int64),
        "opt_attack": torch.zeros(B, O, dtype=torch.int64),
        "opt_scalar": torch.zeros(B, O, F.OPT_SCALAR_DIM),
        "opt_mask": torch.zeros(B, O, dtype=torch.bool),
        "q_type": torch.zeros(B, dtype=torch.int64),
        "q_ctx": torch.zeros(B, dtype=torch.int64),
        "q_scalar": torch.zeros(B, F.Q_SCALAR_DIM),
        "q_ref": torch.full((B, 2), -1, dtype=torch.int64),
        "min_count_t": torch.zeros(B, dtype=torch.int64),
        "max_count_t": torch.zeros(B, dtype=torch.int64),
    }
    for i, s in enumerate(selects):
        o = len(s.opt_type)
        for k, arr in (("opt_type", s.opt_type), ("opt_ref", s.opt_ref),
                       ("opt_ref2", s.opt_ref2),
                       ("opt_card", s.opt_card), ("opt_attack", s.opt_attack)):
            out[k][i, :o] = torch.as_tensor(arr)
        out["opt_scalar"][i, :o] = torch.as_tensor(s.opt_scalar)
        out["opt_mask"][i, :o] = True
        out["q_type"][i], out["q_ctx"][i] = s.q_type, s.q_ctx
        out["q_scalar"][i] = torch.as_tensor(s.q_scalar)
        out["q_ref"][i] = torch.as_tensor(s.q_ref)
        out["min_count_t"][i], out["max_count_t"][i] = s.min_count, s.max_count
    return out


def critic_config(tables):
    return ModelConfig(d=256, layers=4, heads=8, ffn=1024, dec_layers=0,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


class CriticModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None):
        super().__init__()
        self.encoder = Encoder(cfg, attr_table)
        self.head = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(),
                                  nn.Linear(cfg.d, 2), nn.Tanh())
        # Zero-init the final linear (before Tanh) so initial values are
        # exactly 0 and the first optimizer steps cannot overshoot.
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, batch):
        h = self.encoder(batch)
        return self.head(torch.cat([h[:, VALUE_ROWS[0]], h[:, VALUE_ROWS[1]]], -1))
