"""Ship-gate eval: search agent vs raw-policy agent, mirror decks.

Both sides are independent loads of submission_src/main.py (module state
isolated); side A searches, side B has _SEARCH_ENABLED=False. Seats
alternate per game. Workers run whole games sequentially (one battle per
process at a time); the parent aggregates and prints a Wilson 95% CI.

Gate (user-agreed): wr >= 0.55 AND ci_lo > 0.50.

Usage:
  python scripts/eval_search.py --games 400 --workers 3 --cap 0.8 --k 3 --sims 24
  python scripts/eval_search.py --games 20 --workers 1 --full          # full budgets
  python scripts/eval_search.py --games 60 --workers 3 --opp-deck dragapult-ex
"""
import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_main(name):
    path = os.path.join(REPO, "submission_src", "main.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _play_chunk(args):
    (games, seed0, cap, k, sims, full, opp_deck_slug) = args
    from ptcg.engine import BattleSession

    a = _load_main("submission_main_search")     # searcher
    b = _load_main("submission_main_policy")     # raw policy
    b._SEARCH_ENABLED = False
    if not full:
        a._configure_search(cap_s=cap, k_trees=k, sims_per_tree=sims,
                            bank_s=10_000.0)
    deck_a = a.agent({"select": None, "current": None, "logs": []})
    if opp_deck_slug:
        p = os.path.join(REPO, "decks", opp_deck_slug, "deck.csv")
        with open(p) as f:
            b._DECK = [int(r) for r in f.read().split("\n") if r.strip()][:60]
    deck_b = b.agent({"select": None, "current": None, "logs": []})
    wins = 0
    for g in range(games):
        seat_a = g % 2
        a.agent({"select": None, "current": None, "logs": []})
        b.agent({"select": None, "current": None, "logs": []})
        a._reseed(seed0 + 2 * g)
        random.seed(seed0 + 2 * g + 1)
        s = BattleSession(deck_a if seat_a == 0 else deck_b,
                          deck_b if seat_a == 0 else deck_a)
        try:
            while not s.done:
                mod = a if s.select_player == seat_a else b
                picks = mod.agent(s.obs)
                assert mod._is_legal(picks, s.obs["select"]), picks
                s.select(picks)
            if s.result == seat_a:
                wins += 1
        finally:
            s.close()
    return {"wins": wins, "games": games,
            "searched": a._TELEM["searched"], "sims": a._TELEM["sims"],
            "fallbacks": a._TELEM["fallbacks"],
            "search_time": a._TELEM["search_time"]}


def _wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cap", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--full", action="store_true",
                    help="default budgets (no reduced-cap override)")
    ap.add_argument("--opp-deck", default="",
                    help="portfolio deck slug for the raw-policy opponent")
    args = ap.parse_args()

    per = args.games // args.workers
    rem = args.games - per * args.workers
    jobs = []
    for w in range(args.workers):
        n = per + (1 if w < rem else 0)
        if n:
            jobs.append((n, 50_000 + 10_000 * w, args.cap, args.k, args.sims,
                         args.full, args.opp_deck))
    if len(jobs) == 1:
        stats = [_play_chunk(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(_play_chunk, jobs)
    wins = sum(s["wins"] for s in stats)
    games = sum(s["games"] for s in stats)
    lo, hi = _wilson(wins, games)
    wr = wins / max(games, 1)
    searched = sum(s["searched"] for s in stats)
    sims = sum(s["sims"] for s in stats)
    st = sum(s["search_time"] for s in stats)
    fb = sum(s["fallbacks"] for s in stats)
    print(f"search-vs-policy: games={games} wins={wins} wr={wr:.3f} "
          f"wilson95=[{lo:.3f},{hi:.3f}]")
    print(f"search telemetry: searched={searched} sims={sims} "
          f"fallbacks={fb} search_time={st:.0f}s "
          f"({st/max(searched,1):.2f}s/searched-move)")
    gate = wr >= 0.55 and lo > 0.50
    print("GATE:", "PASS" if gate else "FAIL",
          "(need wr>=0.55 and ci_lo>0.50)")


if __name__ == "__main__":
    main()
