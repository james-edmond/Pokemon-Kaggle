"""Measurement 3: multi-process scaling of tier (b) random self-play.

Each worker process: import engine (GameInitialize), 3 warmup games, wait at a
barrier, then run independent games for WINDOW_S seconds. Spawn start method
(Windows/macOS default); worker startup is excluded from the timed window.
Worker counts: powers of two up to logical cores, plus N-1 and N.
"""
import multiprocessing as mp
import random
import sys
import time

import common

WINDOW_S = 30.0
WARMUP_GAMES = 3


def worker(wid, barrier, window_s, q):
    common.add_engine_to_path()
    from cg.sim import lib  # pays GameInitialize once per process
    deck = common.load_deck()
    for i in range(WARMUP_GAMES):
        common.play_game_tier_b(lib, deck, random.Random(90_000 + wid * 100 + i))
    barrier.wait()
    t0 = time.perf_counter()
    games = sels = caps = 0
    while True:
        s, r, capped = common.play_game_tier_b(
            lib, deck, random.Random(100_000 + wid * 10_000 + games))
        games += 1
        sels += s
        caps += capped
        t_last = time.perf_counter()
        if t_last - t0 >= window_s:
            break
    q.put({"wid": wid, "games": games, "sels": sels, "caps": caps,
           "elapsed": t_last - t0, "rss": common.rss_bytes()})


def run_config(n_workers):
    barrier = mp.Barrier(n_workers)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(w, barrier, WINDOW_S, q))
             for w in range(n_workers)]
    for p in procs:
        p.start()
    results = [q.get(timeout=WINDOW_S + 120) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(f"worker exited with code {p.exitcode}")
    return results


def default_counts():
    n = mp.cpu_count()
    counts = []
    k = 1
    while k < n:
        counts.append(k)
        k *= 2
    counts += [n - 1, n]
    return sorted({c for c in counts if c >= 1})


def main():
    counts = [int(x) for x in sys.argv[1:]] or default_counts()
    print(f"logical cores: {mp.cpu_count()}, window {WINDOW_S}s/config, "
          f"worker counts {counts}")
    base_rate = None
    for n in counts:
        results = sorted(run_config(n), key=lambda r: r["wid"])
        rate = sum(r["games"] / r["elapsed"] for r in results)
        sel_rate = sum(r["sels"] / r["elapsed"] for r in results)
        games = sum(r["games"] for r in results)
        caps = sum(r["caps"] for r in results)
        rss_mb = [r["rss"] / 2**20 for r in results]
        if base_rate is None:
            base_rate = rate
        eff = rate / (n * base_rate)
        print(f"workers={n}: {games} games, {rate:.2f} games/s, {sel_rate:.0f} sel/s, "
              f"efficiency {100 * eff:.0f}%, rss/worker MB "
              f"min={min(rss_mb):.0f} mean={sum(rss_mb) / n:.0f} max={max(rss_mb):.0f}"
              + (f", CAP HITS={caps}" if caps else ""), flush=True)


if __name__ == "__main__":
    main()
