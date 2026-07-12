"""Promotion gate: candidate vs incumbent. Pre-registered rule:
PROMOTE iff search-wrapped head-to-head Wilson lo > 0.50 (n>=300 default)
AND raw candidate beats random >= 0.85 AND candidate-with-search beats the
frozen phase-3 anchor-with-search >= 0.55. Workers run whole games
sequentially (one battle per process); spawn pool."""
import argparse
import json
import multiprocessing as mp
import os
import random
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


def _search_chunk(args):
    (games, seed, cand, opp, k, sims) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.mcts import SearchConfig
    from ptcg.selfplay_search import play_search_game, sample_deck_pair
    from ptcg.simsearch import SearchSession
    tables = build_tables()
    a = _load_policy(cand, tables)
    b = _load_policy(opp, tables)
    cfg = SearchConfig(k_trees=k, sims_per_tree=sims)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    session = SearchSession()   # ONE arena per worker (no native free exists)
    wins = 0
    for g in range(games):
        names = sample_deck_pair(rng)
        seat_a = g % 2
        nets = (a, b) if seat_a == 0 else (b, a)
        game = play_search_game(nets[0], nets[1], names, tables, cfg=cfg,
                                rng=rng, gen=gen, record=False,
                                session=session)
        if game.result == seat_a:
            wins += 1
    return {"wins": wins, "games": games}


def _raw_chunk(args):
    (games, seed, cand, opp_spec) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.rollout import play_league_game
    from ptcg.selfplay_search import sample_deck_pair
    from ptcg.decks import deck as deck_by_name
    tables = build_tables()
    a = _load_policy(cand, tables)
    opp = "random" if opp_spec == "random" else _load_policy(opp_spec, tables)
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    wins = 0
    for g in range(games):
        names = sample_deck_pair(rng)
        decks = (deck_by_name(names[0]), deck_by_name(names[1]))
        seat_a = g % 2
        ep = play_league_game(a, opp, decks, tables, learner_seat=seat_a,
                              mirror=False, generator=gen)
        if ep.result == seat_a:
            wins += 1
    return {"wins": wins, "games": games}


def _pool_run(fn, total, workers, mk_args):
    per = total // workers
    rem = total - per * workers
    jobs = [mk_args(w, per + (1 if w < rem else 0)) for w in range(workers)
            if per + (1 if w < rem else 0) > 0]
    if len(jobs) == 1:
        stats = [fn(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(fn, jobs)
    return (sum(s["wins"] for s in stats), sum(s["games"] for s in stats))


def main():
    sys.path.insert(0, REPO)
    from ptcg.ei import wilson
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games-search", type=int, default=300)
    ap.add_argument("--games-raw", type=int, default=60)
    ap.add_argument("--anchor", default="champ/phase3-generalist-r120.pt")
    ap.add_argument("--anchor-games", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    def block(w, n):
        lo, hi = wilson(w, n)
        return {"wins": w, "games": n, "wr": w / max(n, 1),
                "lo": lo, "hi": hi}

    w, n = _pool_run(_search_chunk, a.games_search, a.workers,
                     lambda i, g: (g, a.seed + 100 * i, a.candidate,
                                   a.incumbent, a.k, a.sims))
    search_h2h = block(w, n)
    w, n = _pool_run(_raw_chunk, a.games_raw, a.workers,
                     lambda i, g: (g, a.seed + 1000 + 100 * i, a.candidate,
                                   a.incumbent))
    raw_h2h = block(w, n)
    w, n = _pool_run(_raw_chunk, a.games_raw, a.workers,
                     lambda i, g: (g, a.seed + 2000 + 100 * i, a.candidate,
                                   "random"))
    vs_random = block(w, n)
    anchor = {}
    if a.anchor and os.path.exists(a.anchor):
        w, n = _pool_run(_search_chunk, a.anchor_games, a.workers,
                         lambda i, g: (g, a.seed + 3000 + 100 * i,
                                       a.candidate, a.anchor, a.k, a.sims))
        anchor = block(w, n)
    promote = (search_h2h["lo"] > 0.50
               and vs_random["wr"] >= 0.85
               and (not anchor or anchor["wr"] >= 0.55))
    out = {"search_h2h": search_h2h, "raw_h2h": raw_h2h,
           "vs_random": vs_random, "vs_anchor_search": anchor,
           "promote": promote, "args": vars(a)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("gate_ei:", "PROMOTE" if promote else "REJECT",
          json.dumps({k: v for k, v in out.items() if k != "args"}))


if __name__ == "__main__":
    main()
