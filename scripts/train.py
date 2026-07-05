"""Phase-2 training entry. Run from the repo root with the training venv:
venv-train\\Scripts\\python scripts\\train.py --run-id phase2-a --device cuda
"""
import argparse
from dataclasses import fields

from ptcg.trainloop import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-rounds", type=int, default=1_000_000)
    for f in fields(TrainConfig):
        if f.name == "run_dir":
            continue
        t = type(f.default)
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=t,
                        default=f.default)
    args = ap.parse_args()
    kw = {f.name: getattr(args, f.name) for f in fields(TrainConfig)
          if f.name != "run_dir"}
    cfg = TrainConfig(run_dir=f"runs/{args.run_id}", **kw)
    train(cfg, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
