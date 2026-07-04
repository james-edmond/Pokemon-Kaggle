import json
import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import (
    F_COUNT, KIND_ENTITY, KIND_MULTISET, MAX_TOKENS, OWNER_OPP, OWNER_SELF,
    featurize_privileged, featurize_state)
from ptcg.tracker import BeliefTracker

AREA_HAND_Z = 2  # AreaType.HAND, used as the zone id for hand tokens
AREA_DECK_Z = 1  # AreaType.DECK, used as the zone id for deck tokens


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


def test_privileged_early_steps_source_unseen_hand_from_viz():
    """Before a seat's first selection, its slot in last_obs holds the OTHER
    seat's obs (hand=None). viz_hands must supply that seat's hand tokens and
    remove the matching overcount from its deck∪prize union — exactly the
    seeding play_game uses."""
    tables = build_tables()
    deck = load_sample_deck()
    decks = (deck, list(deck))
    s = BattleSession(deck, list(deck))
    rng = random.Random(9)
    try:
        last_obs = [s.obs, s.obs]
        seen = [False, False]
        checked = False
        while not s.done and not checked:
            me = s.select_player
            last_obs[me] = s.obs
            seen[me] = True
            if seen[0] != seen[1]:
                unseen = seen.index(False)
                vp = s.viz_current()["players"]
                viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
                if viz_hands[unseen]:
                    base = featurize_privileged(last_obs[0], last_obs[1],
                                                decks, tables)
                    pv = featurize_privileged(last_obs[0], last_obs[1], decks,
                                              tables, viz_hands=viz_hands)
                    owner = OWNER_SELF if unseen == 0 else OWNER_OPP

                    def hand_rows(ts):
                        return [i for i in range(ts.n)
                                if ts.zone[i] == AREA_HAND_Z
                                and ts.owner[i] == owner
                                and ts.kind[i] == KIND_ENTITY]

                    def union_total(ts):
                        return round(sum(
                            float(ts.numeric[i, F_COUNT]) / 0.25
                            for i in range(ts.n)
                            if ts.zone[i] == AREA_DECK_Z
                            and ts.owner[i] == owner
                            and ts.kind[i] == KIND_MULTISET))

                    n_hand = len(viz_hands[unseen])
                    assert hand_rows(base) == []          # the old, broken view
                    assert len(hand_rows(pv)) == n_hand   # hand tokens present
                    # union no longer overcounts by the unseen hand's size
                    assert union_total(base) - union_total(pv) == n_hand
                    checked = True
            s.select(random_picks(s.obs, rng))
        assert checked, "never reached a one-seat-seen state with a dealt hand"
    finally:
        s.close()


def test_privileged_viz_exposes_opponent_deck_identities():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    # cg is importable only once the engine has loaded (BattleSession above)
    from cg.game import visualize_data
    rng = random.Random(11)
    try:
        last = [None, None]
        while not s.done and (last[0] is None or last[1] is None):
            last[s.select_player] = s.obs
            s.select(random_picks(s.obs, rng))
        assert last[0] is not None and last[1] is not None
        # viz is the latest VisualizeData snapshot's `current` dict
        viz = json.loads(visualize_data())[-1]["current"]
        decks = (deck, list(deck))
        base = featurize_privileged(last[0], last[1], decks, tables)
        pv = featurize_privileged(last[0], last[1], decks, tables, viz=viz)

        # invariants: valid token state, within budget, mask consistent with n
        for ts in (base, pv):
            assert ts.n == int(ts.mask.sum()) <= MAX_TOKENS

        def opp_deck_entities(ts):
            return [
                i for i in range(ts.n)
                if ts.zone[i] == AREA_DECK_Z and ts.owner[i] == OWNER_OPP
                and ts.kind[i] == KIND_ENTITY
            ]

        # viz exposes per-card opponent deck identities as entity tokens;
        # the viz=None view has none (only aggregate multisets in that zone)
        assert opp_deck_entities(base) == []
        assert len(opp_deck_entities(pv)) > 0
    finally:
        s.close()
