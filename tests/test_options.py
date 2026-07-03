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
