import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_game


def test_untrained_model_plays_legal_games():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        ep = play_game(m, (deck, list(deck)), tables, generator=g)
        assert ep.result in (0, 1, 2)
        assert 5 <= len(ep.steps) <= 1000
        assert all(s.logprob <= 0 and s.logprob == s.logprob for s in ep.steps)
        assert ep.rewards in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
