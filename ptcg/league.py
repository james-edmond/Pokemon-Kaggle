import os
from pathlib import Path

import torch

from .model import PolicyModel
from .trainloop import model_config_for


def league_dir(cfg) -> Path:
    return Path(cfg.run_dir) / "league"


def _snap_path(cfg, r) -> Path:
    return league_dir(cfg) / f"snap-{r:04d}.pt"


def snapshot(cfg, round_n, policy) -> Path:
    d = league_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = _snap_path(cfg, round_n)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({k: v.cpu() for k, v in policy.state_dict().items()}, tmp)
    os.replace(tmp, path)
    return path


def snapshot_rounds(cfg) -> list:
    d = league_dir(cfg)
    if not d.exists():
        return []
    return sorted(int(p.stem.split("-")[1]) for p in d.glob("snap-*.pt"))


def prune_pool(cfg, cap, anchors=2) -> list:
    rounds = snapshot_rounds(cfg)
    if len(rounds) <= cap:
        return rounds
    keep = set(rounds[:anchors]) | set(rounds[-(cap - anchors):])
    for r in rounds:
        if r not in keep:
            _snap_path(cfg, r).unlink()
    return sorted(keep)


def sample_opponent(cfg, round_n, rng):
    pool = snapshot_rounds(cfg)
    u = rng.random()
    if u < cfg.mirror_frac or not pool:
        return ("current", None)
    if u < cfg.mirror_frac + cfg.pool_frac:
        r = rng.choice(pool)
        return ("pool", str(_snap_path(cfg, r)))
    return ("random", None)


def load_opponent(path, tables, cfg) -> PolicyModel:
    p = PolicyModel(model_config_for(cfg.model_size, tables))
    sd = torch.load(path, map_location="cpu", weights_only=False)
    p.load_state_dict(sd)
    p.eval()
    return p
