import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.featurize import KIND_ENTITY, OWNER_OPP
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_game

AREA_DECK_Z = 1  # featurize does not export AreaType ids; literal as in phase-1 tests


def _opp_deck_entities(ts):
    return [i for i in range(ts.n)
            if ts.zone[i] == AREA_DECK_Z and ts.owner[i] == OWNER_OPP
            and ts.kind[i] == KIND_ENTITY]


def test_priv_viz_adds_deck_entities_and_default_unchanged():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(4)
    ep_off = play_game(m, (deck, list(deck)), tables, generator=g)
    assert all(_opp_deck_entities(s.priv_state) == [] for s in ep_off.steps)
    ep_on = play_game(m, (deck, list(deck)), tables, generator=g,
                      priv_viz=True)
    assert any(_opp_deck_entities(s.priv_state) for s in ep_on.steps)
    for s in ep_on.steps:
        assert s.priv_state.n == int(s.priv_state.mask.sum())


def test_obs_log_captures_raw_observations():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(5)
    log = []
    ep = play_game(m, (deck, list(deck)), tables, generator=g, obs_log=log)
    assert len(log) == len(ep.steps)
    assert all(isinstance(o, dict) and "current" in o for o in log)
