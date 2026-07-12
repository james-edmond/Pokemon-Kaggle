"""Train one expert-iteration cycle from recorded search self-play data.
Usage:
  python scripts/train_ei.py --data runs/ei/dev/cycle-0/data \
      --ckpt-in submission_src/policy.pt --ckpt-out runs/ei/dev/cycle-0/ckpt.pt \
      [--replay runs/ei/dev/cycle--1/data] [--replay-ratio 0.25] [--kl-coef 0.02]
Streaming: batch files are loaded one at a time (RAM bounded by one file).
Replay mixing is a file-count proxy for the move ratio: ~replay-ratio *
(fresh file count) replay files (oldest-first as globbed) are appended."""
import argparse
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
    from ptcg.ei import EIConfig, iter_game_files, train_ei_stream
    tables = build_tables()
    policy = _load_policy(a.ckpt_in, tables)
    incumbent = _load_policy(a.ckpt_in, tables) if a.kl_coef > 0 else None
    files = iter_game_files([a.data])
    replay_files = iter_game_files(a.replay) if a.replay else []
    cfg = EIConfig(lr=a.lr, epochs=a.epochs, minibatch=a.minibatch,
                   pi_temp=a.pi_temp, kl_coef=a.kl_coef, device=a.device,
                   seed=a.seed)
    metrics = train_ei_stream(policy, files, tables, cfg,
                              incumbent=incumbent, replay_files=replay_files,
                              replay_ratio=a.replay_ratio)
    os.makedirs(os.path.dirname(a.ckpt_out) or ".", exist_ok=True)
    torch.save({"policy": policy.state_dict(),
                "ei_config": vars(a), "metrics": metrics}, a.ckpt_out)
    print("train_ei:", json.dumps(metrics))


if __name__ == "__main__":
    main()
