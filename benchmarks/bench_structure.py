"""Measurement 1: game structure under random-vs-random self-play.

Single process, documented cg.game API (dict level, no dataclass conversion).
Collects selections/game, turns/game, option-list size distribution,
forced-choice share, result tally, per-game wall time.
"""
import random
import time
from collections import Counter

import common

common.add_engine_to_path()
from cg import game  # noqa: E402

N_WARMUP = 3
N_GAMES = 200
SEED_BASE = 1000


def play_one(seed, opt_sizes, max_counts):
    """Returns (selections, turns, result, capped, wall_seconds)."""
    rng = random.Random(seed)
    deck = DECK
    t0 = time.perf_counter()
    obs, start = game.battle_start(deck, deck)
    if obs is None:
        raise RuntimeError(f"battle_start failed: {start.errorPlayer=} {start.errorType=}")
    selections = 0
    sel_by_player = [0, 0]
    forced = 0
    capped = False
    try:
        while obs["current"]["result"] == -1:
            if selections >= common.SELECTION_CAP:
                capped = True
                break
            sel = obs["select"]
            n, mc = len(sel["option"]), sel["maxCount"]
            sel_by_player[obs["current"]["yourIndex"]] += 1
            if opt_sizes is not None:
                opt_sizes.append(n)
                max_counts[mc] += 1
                if mc == 0 or mc == n:  # C(n, mc) == 1: exactly one legal action
                    forced += 1
            obs = game.battle_select(common.random_action(sel, rng))
            selections += 1
        turns = obs["current"]["turn"]
        result = obs["current"]["result"] if not capped else -999
    finally:
        game.battle_finish()
    wall = time.perf_counter() - t0
    return selections, turns, result, capped, wall, forced, sel_by_player


def main():
    global DECK
    DECK = common.load_deck()

    for i in range(N_WARMUP):
        play_one(900 + i, None, Counter())

    opt_sizes = []
    max_counts = Counter()
    per_game = []
    per_player = []
    per_game_max_player = []
    results = Counter()
    cap_hits = 0
    total_forced = 0
    t_all = time.perf_counter()
    for i in range(N_GAMES):
        s, t, r, capped, w, forced, sbp = play_one(SEED_BASE + i, opt_sizes, max_counts)
        per_game.append((s, t, w))
        per_player.extend(sbp)
        per_game_max_player.append(max(sbp))
        results[r] += 1
        cap_hits += capped
        total_forced += forced
    t_all = time.perf_counter() - t_all

    sels = [g[0] for g in per_game]
    turns = [g[1] for g in per_game]
    walls_ms = [g[2] * 1e3 for g in per_game]

    print(f"games={N_GAMES} warmup={N_WARMUP} seeds={SEED_BASE}..{SEED_BASE + N_GAMES - 1}")
    print(common.fmt_summary("selections/game", sels))
    print(common.fmt_summary("selections/game/player", per_player))
    print(common.fmt_summary("selections/game busier player", per_game_max_player))
    print(common.fmt_summary("turns/game", turns))
    print(common.fmt_summary("game wall time", walls_ms, "ms"))
    print(common.fmt_summary("option-list size (per selection)", opt_sizes))
    dist = Counter(opt_sizes)
    top = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())[:12])
    print(f"option-size counts (first 12 sizes): {top}")
    print(f"maxCount counts (first 8): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(max_counts.items())[:8]))
    n_sel = len(opt_sizes)
    print(f"forced selections (maxCount==0 or ==len(option)): {total_forced}/{n_sel} "
          f"= {100 * total_forced / n_sel:.1f}%")
    print(f"results: P0 wins={results[0]} P1 wins={results[1]} draws={results[2]} "
          f"capped={cap_hits}")
    print(f"total: {sum(sels)} selections in {t_all:.2f}s "
          f"-> {N_GAMES / t_all:.2f} games/s, {sum(sels) / t_all:.0f} selections/s "
          f"(full game.py dict API)")


if __name__ == "__main__":
    main()
