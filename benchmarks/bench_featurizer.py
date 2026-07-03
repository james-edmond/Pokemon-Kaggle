"""Featurizer throughput: µs per featurize_state+encode_select over random games."""
import random
import statistics
import time

from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import encode_select, featurize_state
from ptcg.tracker import BeliefTracker

tables = build_tables()
deck = load_sample_deck()
rng = random.Random(0)
per_call = []
t_end = time.perf_counter() + 30
games = sel = 0
while time.perf_counter() < t_end:
    s = BattleSession(deck, list(deck))
    trackers = [BeliefTracker(0), BeliefTracker(1)]
    try:
        while not s.done:
            me = s.select_player
            trackers[me].update(s.obs.get("logs", []))
            t0 = time.perf_counter()
            ts = featurize_state(s.obs, me, deck, trackers[me].snapshot(), tables)
            encode_select(s.obs, ts, tables)
            per_call.append(time.perf_counter() - t0)
            sel += 1
            s.select(random_picks(s.obs, rng))
    finally:
        s.close()
    games += 1
us = [x * 1e6 for x in per_call]
print(f"games={games} selections={sel}")
print(f"featurize+encode: mean {statistics.mean(us):.0f} us, "
      f"median {statistics.median(us):.0f} us, "
      f"p95 {sorted(us)[int(0.95 * len(us))]:.0f} us")
