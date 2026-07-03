import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import OWNER_OPP, featurize_privileged, featurize_state
from ptcg.tracker import BeliefTracker

AREA_HAND_Z = 2  # AreaType.HAND, used as the zone id for hand tokens


def test_privileged_sees_both_hands():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    rng = random.Random(5)
    try:
        last = [None, None]
        while not s.done and (last[0] is None or last[1] is None):
            last[s.select_player] = s.obs
            s.select(random_picks(s.obs, rng))
        assert last[0] is not None and last[1] is not None
        # privileged view is built from seat 0's perspective: OWNER_OPP == seat 1
        pv = featurize_privileged(last[0], last[1], (deck, list(deck)), tables)
        opp_hand_rows = [
            i for i in range(pv.n)
            if pv.zone[i] == AREA_HAND_Z and pv.owner[i] == OWNER_OPP
        ]
        expected = last[1]["current"]["players"][1]["handCount"]
        assert len(opp_hand_rows) == expected
        # the public view from seat 0 must contain no opponent hand tokens
        pub = featurize_state(last[0], 0, deck, BeliefTracker(0).snapshot(), tables)
        assert [
            i for i in range(pub.n)
            if pub.zone[i] == AREA_HAND_Z and pub.owner[i] == OWNER_OPP
        ] == []
    finally:
        s.close()
