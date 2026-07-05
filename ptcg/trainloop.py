import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .featurize import FEATURIZER_VERSION
from .model import (CriticModel, PolicyModel, critic_config, student_config,
                    teacher_config, tiny_config)


@dataclass
class TrainConfig:
    run_dir: str
    model_size: str = "student"
    games_per_round: int = 192
    actors: int = 3
    epochs: int = 2
    minibatch: int = 512
    lr: float = 3e-4
    clip: float = 0.2
    lam: float = 0.95
    gamma: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    critic_coef: float = 0.5
    aux_coef: float = 0.1
    kl_stop: float = 0.02
    grad_clip: float = 1.0
    ratio_gate: float = 1e-3
    eval_every: int = 5
    eval_games_random: int = 200
    eval_games_ckpt: int = 100
    device: str = "cpu"
    seed: int = 0
    step_cap: int = 5000


def model_config_for(size, tables):
    return {"tiny": tiny_config, "student": student_config,
            "teacher": teacher_config}[size](tables)


def round_dir(cfg, n) -> Path:
    return Path(cfg.run_dir) / "rounds" / str(n)


def checkpoint_path(cfg, n) -> Path:
    return Path(cfg.run_dir) / f"checkpoint-{n:04d}.pt"


def game_seed(cfg, round_n, actor_idx, game_idx) -> int:
    return ((cfg.seed * 1_000_003 + round_n) * 101 + actor_idx) * 10_007 + game_idx


def save_game(path, episode) -> None:
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(episode, tmp)
    os.replace(tmp, path)


def load_round(cfg, n):
    rd = round_dir(cfg, n)
    eps = []
    for f in sorted(rd.glob("*.pt")):
        eps.append(torch.load(f, weights_only=False))
    return eps


def save_checkpoint(cfg, n, policy, critic, optim) -> None:
    path = checkpoint_path(cfg, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "round": n,
        "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
        "optim": optim.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "config": asdict(cfg),
        "featurizer_version": FEATURIZER_VERSION,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, critic, optim=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["policy"])
    critic.load_state_dict(ck["critic"])
    if optim is not None:
        optim.load_state_dict(ck["optim"])
        torch.set_rng_state(ck["torch_rng"])
    return ck["round"]


def latest_checkpoint(cfg):
    best = None
    for f in Path(cfg.run_dir).glob("checkpoint-*.pt"):
        n = int(f.stem.split("-")[1])
        if best is None or n > best[0]:
            best = (n, f)
    return best
