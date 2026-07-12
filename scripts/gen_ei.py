"""Generate expert-iteration self-play data (search vs search, one net).
Usage:
  python scripts/gen_ei.py --ckpt submission_src/policy.pt \
      --out runs/ei/dev/cycle-0/data --games 80 --workers 3 --k 2 --sims 16 --seed 1
Writes worker-<i>-batch-<j>.pt (list[EIGame]) and manifest.json when complete.
Idempotent-ish: refuses to run if manifest.json already exists."""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_policy(path, tables):
    """Bare policy state_dict OR training checkpoint with a 'policy' key."""
    import torch

    from ptcg.model import PolicyModel, student_config
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob
    m = PolicyModel(student_config(tables))
    m.load_state_dict(sd)
    m.eval()
    return m


def _worker(args):
    (wid, games, ckpt, out, k, sims, seed, batch, mirror_frac) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.mcts import SearchConfig
    from ptcg.selfplay_search import play_search_game, sample_deck_pair
    from ptcg.simsearch import SearchSession
    tables = build_tables()
    net = load_policy(ckpt, tables)
    cfg = SearchConfig(k_trees=k, sims_per_tree=sims)
    rng = random.Random(seed * 1000 + wid)
    gen = torch.Generator().manual_seed(seed * 1000 + wid)
    session = SearchSession()   # ONE arena per worker (no native free exists)
    buf, bi, moves, done = [], 0, 0, 0
    for g in range(games):
        names = sample_deck_pair(rng, mirror_frac)
        game = play_search_game(net, net, names, tables, cfg=cfg, rng=rng,
                                gen=gen, session=session)
        buf.append(game)
        moves += len(game.steps)
        done += 1
        if len(buf) >= batch:
            torch.save(buf, os.path.join(out, f"worker-{wid}-batch-{bi}.pt"))
            buf, bi = [], bi + 1
    if buf:
        torch.save(buf, os.path.join(out, f"worker-{wid}-batch-{bi}.pt"))
    return {"games": done, "moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--mirror-frac", type=float, default=0.3)
    a = ap.parse_args()
    mani = os.path.join(a.out, "manifest.json")
    if os.path.exists(mani):
        sys.exit(f"refusing: {mani} already exists (complete run)")
    os.makedirs(a.out, exist_ok=True)
    per = a.games // a.workers
    rem = a.games - per * a.workers
    jobs = [(w, per + (1 if w < rem else 0), a.ckpt, a.out, a.k, a.sims,
             a.seed, a.batch, a.mirror_frac)
            for w in range(a.workers) if per + (1 if w < rem else 0) > 0]
    if len(jobs) == 1:
        stats = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(_worker, jobs)
    total = {"games": sum(s["games"] for s in stats),
             "moves": sum(s["moves"] for s in stats),
             "args": vars(a)}
    with open(mani, "w") as f:
        json.dump(total, f, indent=2)
    print(f"gen_ei: {total['games']} games, {total['moves']} moves -> {a.out}")


if __name__ == "__main__":
    main()
