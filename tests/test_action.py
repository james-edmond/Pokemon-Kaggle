import random
import torch
from ptcg.action import replay_logprob, sample_select
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck
from ptcg.featurize import encode_select, featurize_state
from ptcg.model import PolicyModel, tiny_config
from ptcg.tracker import BeliefTracker


def _one_select():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        me = s.select_player
        ts = featurize_state(s.obs, me, deck, BeliefTracker(me).snapshot(), tables)
        es = encode_select(s.obs, ts, tables)
    finally:
        s.close()
    return tables, ts, es


def test_sample_respects_bounds_and_no_duplicates():
    tables, ts, es = _one_select()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        d = sample_select(m, ts, es, g)
        assert es.min_count <= len(d.picks) <= es.max_count
        assert len(set(d.picks)) == len(d.picks)
        assert all(0 <= p < len(es.opt_type) for p in d.picks)


def test_replay_matches_sample_exactly():
    tables, ts, es = _one_select()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    d = sample_select(m, ts, es, g)
    lp = replay_logprob(m, [ts], [es], [d.picks])
    assert torch.allclose(lp[0], torch.tensor(d.logprob), atol=0, rtol=0)
