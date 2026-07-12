import random

import torch

from ptcg.cards import build_tables
from ptcg.decks import PORTFOLIO
from ptcg.mcts import SearchConfig
from ptcg.selfplay_search import (EIGame, EIStep, play_search_game,
                                  sample_deck_pair)


def _tiny_net(tables):
    import os

    from ptcg.model import PolicyModel, student_config
    m = PolicyModel(student_config(tables))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m.load_state_dict(torch.load(
        os.path.join(repo, "submission_src", "policy.pt"),
        map_location="cpu", weights_only=True))
    m.eval()
    return m


def test_sample_deck_pair_mirror_and_distinct():
    rng = random.Random(0)
    pairs = [sample_deck_pair(rng) for _ in range(200)]
    names = set(PORTFOLIO)
    assert all(a in names and b in names for a, b in pairs)
    mirrors = sum(1 for a, b in pairs if a == b)
    assert 20 <= mirrors <= 120          # ~30% of 200, loose bounds
    assert any(a != b for a, b in pairs)


def test_play_search_game_records_targets_and_outcome():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=2, sims_per_tree=8)
    rng = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=rng, gen=gen)
    assert isinstance(g, EIGame)
    assert g.result in (0, 1, 2)
    assert g.rewards in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
    assert len(g.decks) == 2 and len(g.decks[0]) == 60
    assert len(g.steps) > 0
    searched = [s for s in g.steps if s.actions is not None]
    valueonly = [s for s in g.steps if s.actions is None]
    assert searched, "no move recorded a visit distribution"
    for s in g.steps:
        assert isinstance(s, EIStep)
        assert s.player in (0, 1)
        assert s.priv_state is not None
        if s.actions is not None:
            assert len(s.actions) == len(s.visits) >= 2
            assert all(isinstance(a, tuple) for a in s.actions)
            assert sum(s.visits) > 0
    # aux_targets consumes EIStep unchanged (field-name contract)
    from ptcg.ppo import aux_targets
    opp = [g.decks[1 - s.player] for s in g.steps]
    pd, dl, hd = aux_targets(g.steps, tables, opp)
    assert pd.shape[0] == len(g.steps) and dl.shape[0] == len(g.steps)
    assert valueonly is not None   # trivial/forced moves may or may not occur


def test_play_search_game_record_false_skips_steps():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=1, sims_per_tree=4)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=random.Random(5),
                         gen=torch.Generator().manual_seed(5), record=False)
    assert g.steps == [] and g.result in (0, 1, 2)
