import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import (
    N_OPT_TYPE, encode_select, featurize_state, hash_id,
)
from ptcg.tracker import BeliefTracker


def test_hash_id_buckets():
    assert hash_id(5, 17) == 5
    assert 17 <= hash_id(23, 17) < 17 + 8
    assert hash_id(23, 17) == hash_id(23 + 8, 17)


def test_unknown_option_type_no_crash():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        ts = featurize_state(s.obs, s.select_player, deck,
                             BeliefTracker(s.select_player).snapshot(), tables)
        sel = {**s.obs["select"], "option": [{"type": 9999}]}
        es = encode_select({**s.obs, "select": sel}, ts, tables)
        assert es.opt_type[0] >= 17 and es.opt_type[0] < N_OPT_TYPE
    finally:
        s.close()


# OptionType values (cg/api.py): ATTACH(8)/EVOLVE(9) carry area/index +
# inPlayArea/inPlayIndex and no playerIndex; ABILITY(10)/DISCARD(11) carry
# area/index and no playerIndex. All reference the selecting player's cards.
OPT_ATTACH_T, OPT_EVOLVE_T, OPT_ABILITY_T, OPT_DISCARD_T = 8, 9, 10, 11
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5


def test_attach_evolve_options_encode_distinct():
    """Selects with >=2 ATTACH (or EVOLVE) options must not encode them all
    identically when their (index, inPlayArea, inPlayIndex) differ."""
    tables = build_tables()
    deck = load_sample_deck()
    rng = random.Random(23)
    groups_checked = 0
    for _ in range(5):
        s = BattleSession(deck, list(deck))
        trackers = [BeliefTracker(0), BeliefTracker(1)]
        try:
            while not s.done:
                me = s.select_player
                trackers[me].update(s.obs.get("logs", []))
                opts = s.obs["select"]["option"]
                by_type = {}
                for i, o in enumerate(opts):
                    if o.get("type") in (OPT_ATTACH_T, OPT_EVOLVE_T):
                        by_type.setdefault(o["type"], []).append(i)
                if any(len(v) >= 2 for v in by_type.values()):
                    ts = featurize_state(s.obs, me, deck,
                                         trackers[me].snapshot(), tables)
                    es = encode_select(s.obs, ts, tables)
                    for idxs in by_type.values():
                        if len(idxs) < 2:
                            continue
                        srcs = {(opts[i].get("index"), opts[i].get("inPlayArea"),
                                 opts[i].get("inPlayIndex")) for i in idxs}
                        encs = {(int(es.opt_ref[i]), int(es.opt_ref2[i]))
                                for i in idxs}
                        if len(srcs) > 1:
                            assert len(encs) > 1, \
                                "distinct attach/evolve choices encode identically"
                            groups_checked += 1
                        for i in idxs:
                            if opts[i].get("area") == AREA_HAND:
                                assert es.opt_ref[i] >= 0
                            if opts[i].get("inPlayArea") in (AREA_ACTIVE, AREA_BENCH):
                                assert es.opt_ref2[i] >= 0
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
        if groups_checked:
            break
    assert groups_checked > 0, "sweep never hit a multi-ATTACH/EVOLVE select"


def test_playerless_option_types_resolve_synthetic():
    """ABILITY/DISCARD (and ATTACH/EVOLVE) options resolve to the selecting
    player's own tokens despite carrying no playerIndex. Option shapes match
    real engine probes (ATTACH: area/index/inPlayArea/inPlayIndex)."""
    from test_featurize import _mini_obs
    tables = build_tables()
    deck = load_sample_deck()
    obs = _mini_obs()
    ts = featurize_state(obs, 0, deck, BeliefTracker(0).snapshot(), tables)
    active_row = ts.ref[(0, AREA_ACTIVE, 0, -1)]
    hand0 = ts.ref[(0, AREA_HAND, 0, -1)]
    hand1 = ts.ref[(0, AREA_HAND, 1, -1)]
    options = [
        {"type": OPT_ATTACH_T, "area": AREA_HAND, "index": 0,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OPT_ATTACH_T, "area": AREA_HAND, "index": 1,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OPT_EVOLVE_T, "area": AREA_HAND, "index": 1,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OPT_ABILITY_T, "area": AREA_ACTIVE, "index": 0},
        {"type": OPT_DISCARD_T, "area": AREA_ACTIVE, "index": 0},
    ]
    sel = {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
           "remainDamageCounter": 0, "remainEnergyCost": 0,
           "option": options, "deck": None, "contextCard": None, "effect": None}
    es = encode_select({**obs, "select": sel}, ts, tables)
    assert es.opt_ref[0] == hand0 and es.opt_ref2[0] == active_row
    assert es.opt_ref[1] == hand1 and es.opt_ref2[1] == active_row
    assert es.opt_ref[2] == hand1 and es.opt_ref2[2] == active_row
    assert es.opt_ref[3] == active_row and es.opt_ref2[3] == -1
    assert es.opt_ref[4] == active_row and es.opt_ref2[4] == -1
    assert (int(es.opt_ref[0]), int(es.opt_ref2[0])) != \
           (int(es.opt_ref[1]), int(es.opt_ref2[1]))


def test_real_game_option_sweep():
    tables = build_tables()
    deck = load_sample_deck()
    rng = random.Random(11)
    for _ in range(5):
        s = BattleSession(deck, list(deck))
        trackers = [BeliefTracker(0), BeliefTracker(1)]
        try:
            while not s.done:
                me = s.select_player
                trackers[me].update(s.obs.get("logs", []))
                ts = featurize_state(s.obs, me, deck,
                                     trackers[me].snapshot(), tables)
                es = encode_select(s.obs, ts, tables)
                o = len(s.obs["select"]["option"])
                assert es.opt_type.shape == (o,)
                assert es.min_count <= es.max_count <= o
                assert (es.opt_ref < ts.n).all()
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
