"""Train one expert-iteration cycle from recorded search self-play data.
Usage:
  python scripts/train_ei.py --data runs/ei/dev/cycle-0/data \
      --ckpt-in submission_src/policy.pt --ckpt-out runs/ei/dev/cycle-0/ckpt.pt \
      [--replay runs/ei/dev/cycle--1/data] [--replay-ratio 0.25] [--kl-coef 0.02]
Replay mixing: appends whole games from --replay dirs (oldest-first order
as globbed) until replay moves ~= replay-ratio * fresh moves."""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_policy(path, tables):
    import torch

    from ptcg.model import PolicyModel, student_config
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob
    m = PolicyModel(student_config(tables))
    m.load_state_dict(sd)
    m.eval()
    return m


def _load_games(d):
    import torch
    out = []
    for f in sorted(glob.glob(os.path.join(d, "worker-*.pt"))):
        out.extend(torch.load(f, weights_only=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay", action="append", default=[])
    ap.add_argument("--replay-ratio", type=float, default=0.25)
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--pi-temp", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.ckpt_out):
        sys.exit(f"refusing: {a.ckpt_out} already exists")
    import torch

    from ptcg.cards import build_tables
    from ptcg.ei import EIConfig, train_ei
    tables = build_tables()
    policy = _load_policy(a.ckpt_in, tables)
    incumbent = _load_policy(a.ckpt_in, tables) if a.kl_coef > 0 else None
    games = _load_games(a.data)
    fresh_moves = sum(len(g.steps) for g in games)
    target = int(fresh_moves * a.replay_ratio)
    got = 0
    for rd in a.replay:
        for g in _load_games(rd):
            if got >= target:
                break
            games.append(g)
            got += len(g.steps)
    cfg = EIConfig(lr=a.lr, epochs=a.epochs, minibatch=a.minibatch,
                   pi_temp=a.pi_temp, kl_coef=a.kl_coef, device=a.device,
                   seed=a.seed)
    metrics = train_ei(policy, games, tables, cfg, incumbent=incumbent)
    os.makedirs(os.path.dirname(a.ckpt_out) or ".", exist_ok=True)
    torch.save({"policy": policy.state_dict(),
                "ei_config": vars(a), "metrics": metrics}, a.ckpt_out)
    print("train_ei:", json.dumps(metrics))


if __name__ == "__main__":
    main()
