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
