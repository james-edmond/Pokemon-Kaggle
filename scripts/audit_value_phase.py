"""Non-gating: the turn-boundary value-jump statistic (phase-4 diagnosis).
For each turn transition in random-play games, consistency demands
v(first select of new turn, acting seat) ~= -v(last select of prev turn,
prev seat); the jump is |v_new + v_prev|. Expert iteration should shrink
the mean jump across cycles.
Usage: python scripts/audit_value_phase.py --ckpt <path> --out <json> [--games 6]"""
import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    import numpy as np
    import torch

    from ptcg.cards import build_tables
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.featurize import featurize_state
    from ptcg.model import collate_states
    from ptcg.tracker import BeliefTracker
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gate_ei import _load_policy
    tables = build_tables()
    net = _load_policy(a.ckpt, tables)
    deck = load_sample_deck()
    jumps, all_v = [], []
    for g in range(a.games):
        s = BattleSession(deck, deck)
        try:
            rng = random.Random(a.seed + g)
            trk = {0: BeliefTracker(0), 1: BeliefTracker(1)}
            prev_v = prev_turn = None
            for _ in range(120):
                if s.done:
                    break
                seat = s.obs["current"]["yourIndex"]
                turn = s.obs["current"]["turn"]
                trk[seat].update(s.obs.get("logs") or [])
                ts = featurize_state(s.obs, seat, deck, trk[seat].snapshot(),
                                     tables)
                with torch.no_grad():
                    v = float(net.public_value(net.encode(
                        collate_states([ts]))))
                all_v.append(v)
                if prev_v is not None and prev_turn is not None \
                        and turn != prev_turn:
                    jumps.append(abs(v + prev_v))
                prev_v, prev_turn = v, turn
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
    out = {"mean_jump": float(np.mean(jumps)) if jumps else None,
           "n_jumps": len(jumps), "mean_abs_v": float(np.mean(np.abs(all_v))),
           "games": a.games}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("audit_value_phase:", json.dumps(out))


if __name__ == "__main__":
    main()
