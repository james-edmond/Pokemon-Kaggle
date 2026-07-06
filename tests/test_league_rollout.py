import torch
from ptcg.cards import build_tables
from ptcg.decks import deck as get_deck, all_decks
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_league_game


def test_non_mirror_collects_only_learner_seat():
    tables = build_tables()
    deck = load_sample_deck()
    learner = PolicyModel(tiny_config(tables))
    opp = PolicyModel(tiny_config(tables))     # different weights (a snapshot stand-in)
    g = torch.Generator().manual_seed(0)
    ep = play_league_game(learner, opp, (deck, list(deck)), tables,
                          learner_seat=0, mirror=False, generator=g)
    assert ep.collected_seats == (0,)
    assert all(s.player == 0 for s in ep.steps)      # only learner seat collected
    assert 5 <= len(ep.steps) <= 1000
    assert ep.result in (0, 1, 2)
    assert ep.decks[0] == deck and ep.decks[1] == list(deck)
    assert all(s.logprob <= 0 and s.logprob == s.logprob for s in ep.steps)


def test_mirror_collects_both_seats():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    ep = play_league_game(m, m, (deck, list(deck)), tables,
                          learner_seat=0, mirror=True, generator=g)
    assert ep.collected_seats == (0, 1)
    assert {s.player for s in ep.steps} <= {0, 1}
    assert any(s.player == 1 for s in ep.steps)      # both seats present


def test_learner_seat_1_and_random_opponent():
    tables = build_tables()
    deck = load_sample_deck()
    learner = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(2)
    ep = play_league_game(learner, "random", (deck, list(deck)), tables,
                          learner_seat=1, mirror=False, generator=g)
    assert ep.collected_seats == (1,) and all(s.player == 1 for s in ep.steps)
