"""Measurement 2: single-process step cost, decomposed into tiers.

Part 1 — instrumented decomposition (per-component accumulators around the raw
ctypes calls). Components per selection:
  select     lib.Select                     (C engine: apply action)
  getdata    lib.GetBattleData              (C engine: serialize state)
  json       json.loads(sd.json)            (c_char_p->bytes copy + parse)
  sbi        ctypes.string_at(sd.data)+decode (search_begin_input; game.py always pays this)
  dataclass  cg.api.to_observation_class    (recursive dict->dataclass)
Tiers: (a) = select+getdata, (b) = a+json, (c) = b+dataclass.
Instrumentation overhead: 6 perf_counter calls per selection (~0.1-0.5 us total).

Part 2 — clean tier (b) throughput window (no per-step instrumentation):
raw lib.Select + lib.GetBattleData + json.loads, the realistic floor for an
RL loop with a lean custom parser.
"""
import ctypes
import json
import random
import time

import common

common.add_engine_to_path()
from cg.sim import lib  # noqa: E402
from cg.api import to_observation_class  # noqa: E402

N_WARMUP = 3
N_GAMES_DECOMP = 100
SEED_DECOMP = 2000
SEED_CLEAN = 3000
CLEAN_MIN_S = 30.0
CLEAN_MIN_GAMES = 30


def decomposed_game(seed, acc):
    rng = random.Random(seed)
    cards = DECK + DECK
    arr = (ctypes.c_int * len(cards))(*cards)
    start = lib.BattleStart(arr)
    ptr = start.battlePtr
    if not ptr:
        raise RuntimeError(f"BattleStart failed: {start.errorPlayer=} {start.errorType=}")
    steps = 0
    try:
        obs = json.loads(lib.GetBattleData(ptr).json)
        while obs["current"]["result"] == -1:
            if steps >= common.SELECTION_CAP:
                raise RuntimeError(f"selection cap hit at seed {seed}")
            action = common.random_action(obs["select"], rng)
            sel_arr = (ctypes.c_int * len(action))(*action)
            t0 = time.perf_counter()
            err = lib.Select(ptr, sel_arr, len(action))
            t1 = time.perf_counter()
            if err != 0:
                raise RuntimeError(f"lib.Select error {err} at step {steps}, seed {seed}")
            sd = lib.GetBattleData(ptr)
            t2 = time.perf_counter()
            raw = sd.json
            obs = json.loads(raw)
            t3 = time.perf_counter()
            _sbi = ctypes.string_at(sd.data, sd.count).decode("ascii")
            t4 = time.perf_counter()
            _dc = to_observation_class(obs)
            t5 = time.perf_counter()
            acc["select"] += t1 - t0
            acc["getdata"] += t2 - t1
            acc["json"] += t3 - t2
            acc["sbi"] += t4 - t3
            acc["dataclass"] += t5 - t4
            acc["json_bytes"] += len(raw)
            acc["steps"] += 1
            steps += 1
    finally:
        lib.BattleFinish(ptr)


def main():
    global DECK
    DECK = common.load_deck()

    # Part 1: decomposition
    for i in range(N_WARMUP):
        common.play_game_tier_b(lib, DECK, random.Random(1900 + i))
    acc = {"select": 0.0, "getdata": 0.0, "json": 0.0, "sbi": 0.0, "dataclass": 0.0,
           "steps": 0, "json_bytes": 0}
    t0 = time.perf_counter()
    for i in range(N_GAMES_DECOMP):
        decomposed_game(SEED_DECOMP + i, acc)
    wall = time.perf_counter() - t0

    n = acc["steps"]
    comps = ["select", "getdata", "json", "sbi", "dataclass"]
    total = sum(acc[c] for c in comps)
    print(f"decomposition: {N_GAMES_DECOMP} games, {n} selections, wall {wall:.2f}s "
          f"(timed components cover {100 * total / wall:.1f}% of wall)")
    for c in comps:
        print(f"  {c:9s} {1e6 * acc[c] / n:8.1f} us/sel  {100 * acc[c] / total:5.1f}%")
    tier_a = acc["select"] + acc["getdata"]
    tier_b = tier_a + acc["json"]
    tier_c = tier_b + acc["dataclass"]
    full = total
    print(f"  tier (a) raw engine       {1e6 * tier_a / n:8.1f} us/sel")
    print(f"  tier (b) + json.loads     {1e6 * tier_b / n:8.1f} us/sel")
    print(f"  tier (c) + dataclass      {1e6 * tier_c / n:8.1f} us/sel")
    print(f"  full sample-agent path    {1e6 * full / n:8.1f} us/sel (tier c + sbi)")
    print(f"  mean observation JSON size: {acc['json_bytes'] / n / 1024:.1f} KiB")

    # Part 2: clean tier (b) throughput
    for i in range(N_WARMUP):
        common.play_game_tier_b(lib, DECK, random.Random(2900 + i))
    games = sels = 0
    t0 = time.perf_counter()
    while True:
        s, r, capped = common.play_game_tier_b(lib, DECK, random.Random(SEED_CLEAN + games))
        if capped:
            print(f"*** CAP HIT in clean run, seed {SEED_CLEAN + games} ***")
        games += 1
        sels += s
        elapsed = time.perf_counter() - t0
        if elapsed >= CLEAN_MIN_S and games >= CLEAN_MIN_GAMES:
            break
    print(f"clean tier (b): {games} games, {sels} selections in {elapsed:.2f}s")
    print(f"  {games / elapsed:.2f} games/s, {sels / elapsed:.0f} selections/s, "
          f"{1e6 * elapsed / sels:.1f} us/selection (incl. Python action sampling)")


if __name__ == "__main__":
    main()
