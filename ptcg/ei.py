"""Expert-iteration training: supervised losses from search self-play.

Policy: cross-entropy to the normalized root visit distribution (single-
pick selects fuse policy+value+aux on one collated trunk; multi-pick
candidates replay through batched_replay). Value: MSE to the game outcome
from the acting seat, on EVERY state including turn-starts (recalibrates
the phase-4-diagnosed turn-phase artifact). Aux heads keep their phase-2
Poisson/MSE targets — the determinizer's accuracy is a search input.
"""
import math
import random
from dataclasses import dataclass

import torch

from .ppo import aux_targets


def wilson(w, n, z=1.96):
    """Wilson 95% score interval (lo, hi) for w wins of n games."""
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


@dataclass
class EIConfig:
    lr: float = 2e-4
    epochs: int = 2
    minibatch: int = 128
    pi_temp: float = 1.0
    kl_coef: float = 0.0
    vf_coef: float = 1.0
    aux_coef: float = 0.1
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 0


def flatten_games(games):
    """[(step, z_for_acting_seat, opponent_60_card_deck), ...]"""
    out = []
    for g in games:
        for s in g.steps:
            out.append((s, float(g.rewards[s.player]),
                        list(g.decks[1 - s.player])))
    return out


def is_single_pick(step):
    return all(len(a) <= 1 for a in step.actions)


def pi_targets_single(steps, O, temp):
    """[B, O+1] normalized visit targets; () maps to the done column O."""
    t = torch.zeros(len(steps), O + 1)
    for i, s in enumerate(steps):
        w = torch.tensor([float(v) for v in s.visits])
        if temp != 1.0:
            w = w.clamp(min=1e-9) ** (1.0 / temp)
        w = w / w.sum()
        for a, p in zip(s.actions, w.tolist()):
            col = a[0] if len(a) == 1 else O
            t[i, col] += p
    return t


def _aux_loss(policy, trunk, steps, opp_decks, tables, device):
    poiss = torch.nn.PoissonNLLLoss(log_input=False, full=False)
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_decks)
    pd_t, dl_t, hd_t = pd_t.to(device), dl_t.to(device), hd_t.to(device)
    return (((policy.prize_diff(trunk) - pd_t) ** 2).mean()
            + poiss(policy.aux_decklist(trunk), dl_t)
            + poiss(policy.aux_hand(trunk), hd_t))


def single_pick_loss(policy, steps, zs, opp_decks, tables, cfg,
                     incumbent=None):
    """Fused policy CE + value MSE + aux losses on one collated batch."""
    from .model import collate_selects, collate_states
    dev = torch.device(cfg.device)
    sb = {k: v.to(dev) for k, v in collate_states(
        [s.state for s in steps]).items()}
    selb = {k: v.to(dev) for k, v in collate_selects(
        [s.esel for s in steps]).items()}
    B, O = selb["opt_type"].shape
    picked = torch.zeros((B, O + 1), dtype=torch.bool, device=dev)
    trunk = policy.encode(sb)
    logits = policy.option_logits(trunk, sb, selb, picked)
    logp = torch.log_softmax(logits, dim=-1)
    targets = pi_targets_single(steps, O, cfg.pi_temp).to(dev)
    # candidates are legal by construction, so target mass never sits on a
    # -inf column; nan_to_num guards the 0 * -inf corner on masked columns
    loss_pi = -(targets * torch.nan_to_num(logp, neginf=0.0)).sum(-1).mean()
    v = policy.public_value(trunk)
    z_t = torch.tensor(zs, dtype=torch.float32, device=dev)
    loss_v = ((v - z_t) ** 2).mean()
    loss_aux = _aux_loss(policy, trunk, steps, opp_decks, tables, dev)
    parts = {"loss_pi": float(loss_pi), "loss_v": float(loss_v),
             "loss_aux": float(loss_aux)}
    loss = loss_pi + cfg.vf_coef * loss_v + cfg.aux_coef * loss_aux
    if incumbent is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            it = incumbent.encode(sb)
            il = incumbent.option_logits(it, sb, selb, picked)
            q = torch.softmax(il, dim=-1)
        p = torch.softmax(logits, dim=-1)
        kl = (p * (torch.nan_to_num(torch.log(p.clamp(min=1e-9)), neginf=0.0)
                   - torch.nan_to_num(torch.log(q.clamp(min=1e-9)),
                                      neginf=0.0))).sum(-1).mean()
        loss = loss + cfg.kl_coef * kl
        parts["kl"] = float(kl)
    return loss, parts


def multi_pick_loss(policy, step, z, opp_deck, tables, cfg):
    """-sum(pi_a * logprob(sequence a)) for one multi-pick state, plus
    value/aux on the same trunk (row 0)."""
    from .model import collate_selects, collate_states
    from .replay import batched_replay
    dev = torch.device(cfg.device)
    n = len(step.actions)
    sb = {k: v.to(dev) for k, v in collate_states(
        [step.state] * n).items()}
    selb = {k: v.to(dev) for k, v in collate_selects(
        [step.esel] * n).items()}
    trunk = policy.encode(sb)
    logp, _ = batched_replay(policy, trunk, sb, selb,
                             [list(a) for a in step.actions])
    w = torch.tensor([float(v) for v in step.visits], device=dev)
    if cfg.pi_temp != 1.0:
        w = w.clamp(min=1e-9) ** (1.0 / cfg.pi_temp)
    pi = w / w.sum()
    loss_pi = -(pi * logp).sum()
    v = policy.public_value(trunk[0:1])
    loss_v = (v - torch.tensor([z], device=dev)) ** 2
    loss_aux = _aux_loss(policy, trunk[0:1], [step], [opp_deck], tables, dev)
    loss = loss_pi + cfg.vf_coef * loss_v.mean() + cfg.aux_coef * loss_aux
    return loss, {"loss_pi": float(loss_pi), "loss_v": float(loss_v.mean()),
                  "loss_aux": float(loss_aux)}


def value_only_loss(policy, steps, zs, opp_decks, tables, cfg):
    """Value + aux losses for states without a policy target."""
    from .model import collate_states
    dev = torch.device(cfg.device)
    sb = {k: v.to(dev) for k, v in collate_states(
        [s.state for s in steps]).items()}
    trunk = policy.encode(sb)
    v = policy.public_value(trunk)
    z_t = torch.tensor(zs, dtype=torch.float32, device=dev)
    loss_v = ((v - z_t) ** 2).mean()
    loss_aux = _aux_loss(policy, trunk, steps, opp_decks, tables, dev)
    loss = cfg.vf_coef * loss_v + cfg.aux_coef * loss_aux
    return loss, {"loss_v": float(loss_v), "loss_aux": float(loss_aux)}


def train_ei(policy, games, tables, cfg, incumbent=None):
    """One training pass (cfg.epochs) over the games. Returns metrics."""
    dev = torch.device(cfg.device)
    policy.to(dev)
    policy.train()
    if incumbent is not None:
        incumbent.to(dev)
        incumbent.eval()
    flat = flatten_games(games)
    singles = [(s, z, od) for s, z, od in flat
               if s.actions is not None and is_single_pick(s)]
    multis = [(s, z, od) for s, z, od in flat
              if s.actions is not None and not is_single_pick(s)]
    vonly = [(s, z, od) for s, z, od in flat if s.actions is None]
    rng = random.Random(cfg.seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    agg = {"loss_pi": 0.0, "loss_v": 0.0, "loss_aux": 0.0}
    n_pi = n_va = 0
    for _ in range(cfg.epochs):
        rng.shuffle(singles)
        rng.shuffle(vonly)
        for lo in range(0, len(singles), cfg.minibatch):
            batch = singles[lo:lo + cfg.minibatch]
            loss, parts = single_pick_loss(
                policy, [b[0] for b in batch], [b[1] for b in batch],
                [b[2] for b in batch], tables, cfg, incumbent=incumbent)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_pi"] += parts["loss_pi"]
            agg["loss_v"] += parts["loss_v"]
            agg["loss_aux"] += parts["loss_aux"]
            n_pi += 1
            n_va += 1
        for s, z, od in multis:
            loss, parts = multi_pick_loss(policy, s, z, od, tables, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_pi"] += parts["loss_pi"]
            n_pi += 1
        for lo in range(0, len(vonly), cfg.minibatch):
            batch = vonly[lo:lo + cfg.minibatch]
            loss, parts = value_only_loss(
                policy, [b[0] for b in batch], [b[1] for b in batch],
                [b[2] for b in batch], tables, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_v"] += parts["loss_v"]
            agg["loss_aux"] += parts["loss_aux"]
            n_va += 1
    policy.eval()
    return {"loss_pi": agg["loss_pi"] / max(n_pi, 1),
            "loss_v": agg["loss_v"] / max(n_va, 1),
            "loss_aux": agg["loss_aux"] / max(n_va, 1),
            "n_single": len(singles), "n_multi": len(multis),
            "n_valueonly": len(vonly), "epochs_ran": cfg.epochs}
