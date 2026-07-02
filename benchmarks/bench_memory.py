"""Measurement 4: memory — RSS after startup and growth over 300 sequential
games (leak check). battle_finish is called after every game (in
play_game_tier_b); RSS sampled every 50 games; least-squares slope reported.
"""
import random
import sys
import time

import common

common.add_engine_to_path()
from cg.sim import lib  # noqa: E402

N_GAMES = 300
SAMPLE_EVERY = 50
SEED_BASE = 4000


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else N_GAMES
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE_EVERY
    deck = common.load_deck()
    print(f"rss after import+GameInitialize: {common.rss_bytes() / 2**20:.1f} MiB")
    for i in range(3):
        common.play_game_tier_b(lib, deck, random.Random(3900 + i))
    baseline = common.rss_bytes()
    print(f"rss after 3 warmup games:        {baseline / 2**20:.1f} MiB")

    samples = [(0, baseline)]
    t0 = time.perf_counter()
    for i in range(n_games):
        s, r, capped = common.play_game_tier_b(lib, deck, random.Random(SEED_BASE + i))
        if capped:
            print(f"*** CAP HIT at seed {SEED_BASE + i} ***")
        if (i + 1) % sample_every == 0:
            samples.append((i + 1, common.rss_bytes()))
    wall = time.perf_counter() - t0

    for g, r in samples:
        print(f"  after {g:4d} games: rss {r / 2**20:7.2f} MiB ({(r - baseline) / 1024:+.0f} KiB)")
    # least-squares slope in bytes/game over the samples
    n = len(samples)
    mx = sum(g for g, _ in samples) / n
    my = sum(r for _, r in samples) / n
    denom = sum((g - mx) ** 2 for g, _ in samples)
    slope = sum((g - mx) * (r - my) for g, r in samples) / denom
    print(f"{n_games} games in {wall:.1f}s; rss slope {slope:+.0f} bytes/game "
          f"({slope * 1000 / 2**20:+.2f} MiB per 1000 games)")


if __name__ == "__main__":
    main()
