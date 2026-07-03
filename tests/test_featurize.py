import numpy as np
import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import (
    MAX_TOKENS, F_COUNT, KIND_CHILD, SUB_ENERGY, featurize_state,
)
from ptcg.tracker import BeliefTracker


def _mini_obs(me=0):
    def pkm(cid, serial, energy_cid):
        return {"id": cid, "serial": serial, "hp": 100, "maxHp": 100,
                "appearThisTurn": False, "energies": [1],
                "energyCards": [{"id": energy_cid, "serial": serial + 500,
                                 "playerIndex": 0}],
                "tools": [], "preEvolution": []}
    def player(hand):
        return {"active": [pkm(745, 1, 1)], "bench": [], "benchMax": 5,
                "deckCount": 50, "discard": [], "prize": [None] * 6,
                "handCount": len(hand) if hand else 0, "hand": hand,
                "poisoned": False,
                "burned": False, "asleep": False, "paralyzed": False,
                "confused": False}
    hand = [{"id": 746, "serial": 10, "playerIndex": 0},
            {"id": 747, "serial": 11, "playerIndex": 0}]
    p1 = player(hand)
    p2 = player(None)
    p2["hand"] = None
    return {"select": None, "logs": [],
            "current": {"turn": 3, "turnActionCount": 0, "yourIndex": me,
                        "firstPlayer": 0, "supporterPlayed": False,
                        "stadiumPlayed": False, "energyAttached": False,
                        "retreated": False, "result": -1, "stadium": [],
                        "looking": None, "players": [p1, p2]}}


def test_mini_obs_refs_and_children():
    tables = build_tables()
    deck = load_sample_deck()
    ts = featurize_state(_mini_obs(), 0, deck, BeliefTracker(0).snapshot(), tables)
    assert (0, 4, 0, -1) in ts.ref            # my active
    assert (0, 4, 0, SUB_ENERGY + 0) in ts.ref  # its attached energy
    child_row = ts.ref[(0, 4, 0, SUB_ENERGY + 0)]
    assert ts.kind[child_row] == KIND_CHILD
    assert ts.n == int(ts.mask.sum()) <= MAX_TOKENS


def test_union_multiset_decrements():
    tables = build_tables()
    deck = load_sample_deck()
    ts = featurize_state(_mini_obs(), 0, deck, BeliefTracker(0).snapshot(), tables)
    # deck.csv holds 4 copies of 722; none visible in mini obs -> count 4/4
    row = ts.mrow[(0, 1, 722)]
    assert np.isclose(ts.numeric[row, F_COUNT], 1.0)


def test_real_game_sweep():
    tables = build_tables()
    deck = load_sample_deck()
    rng = random.Random(3)
    for _ in range(3):
        s = BattleSession(deck, list(deck))
        trackers = [BeliefTracker(0), BeliefTracker(1)]
        try:
            while not s.done:
                me = s.select_player
                trackers[me].update(s.obs.get("logs", []))
                ts = featurize_state(s.obs, me, deck,
                                     trackers[me].snapshot(), tables)
                assert ts.n == int(ts.mask.sum()) <= MAX_TOKENS
                cur = s.obs["current"]
                for pi, pl in enumerate(cur["players"]):
                    for area, lst in ((4, pl["active"]), (5, pl["bench"])):
                        for i, pk in enumerate(lst):
                            if pk is not None:
                                assert (pi, area, i, -1) in ts.ref
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
