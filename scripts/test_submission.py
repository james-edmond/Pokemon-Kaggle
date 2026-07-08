"""End-to-end smoke: the submission agent drives one seat of real BattleSession games.
Proves obs-schema compatibility + legality + latency. Run from repo root (base python),
one engine process at a time. Usage: python scripts/test_submission.py [n_games]"""
import importlib.util
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ptcg.engine import BattleSession, load_sample_deck, random_picks


def load_agent():
    path = os.path.join(REPO, "submission_src", "main.py")
    spec = importlib.util.spec_from_file_location("submission_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(mod, my_seat, my_deck, opp_deck, seed):
    rng = random.Random(seed)
    s = BattleSession(my_deck if my_seat == 0 else opp_deck,
                      opp_deck if my_seat == 0 else my_deck)
    lat = []
    try:
        while not s.done:
            me = s.select_player
            if me == my_seat:
                t0 = time.perf_counter()
                picks = mod.agent(s.obs)
                lat.append(time.perf_counter() - t0)
                assert mod._is_legal(picks, s.obs["select"]), (picks, s.obs["select"])
                s.select(picks)
            else:
                s.select(random_picks(s.obs, rng))
        return s.result, lat
    finally:
        s.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    mod = load_agent()
    # The agent's OWN deck is what it declares at deck selection; the engine deals it
    # that deck, so the smoke must play my_deck = the declared deck (NOT the sample deck),
    # or featurization would mismatch the dealt cards once deck.csv changes in Task 3.
    my_deck = mod.agent({"select": None, "current": None, "logs": []})
    assert len(my_deck) == 60
    opp_deck = load_sample_deck()   # a fixed opponent deck for the smoke
    wins, done, all_lat = 0, 0, []
    for g in range(n):
        my_seat = g % 2
        mod.agent({"select": None, "current": None, "logs": []})  # reset per-game state
        result, lat = play(mod, my_seat, my_deck, opp_deck, seed=1000 + g)
        done += 1
        if result == my_seat:
            wins += 1
        all_lat += lat
    all_lat.sort()
    p50 = all_lat[len(all_lat) // 2]
    p95 = all_lat[int(len(all_lat) * 0.95)]
    print(f"games={done} wins={wins} winrate={wins/done:.3f} "
          f"moves={len(all_lat)} latency p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms")
    print("OK — all agent picks legal, all games completed" if done == n else "INCOMPLETE")


if __name__ == "__main__":
    main()
