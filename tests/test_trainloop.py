import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.rollout import play_game
from ptcg.trainloop import (TrainConfig, checkpoint_path, latest_checkpoint,
                            load_checkpoint, load_round, round_dir, save_checkpoint,
                            save_game)


def test_game_roundtrip(tmp_path):
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    ep = play_game(m, (deck, list(deck)), tables, generator=g)
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    rd = round_dir(cfg, 0)
    rd.mkdir(parents=True)
    save_game(rd / "a0-g0.pt", ep)
    (eps,) = (load_round(cfg, 0),)
    assert len(eps) == 1
    e = eps[0]
    assert e.result == ep.result and e.rewards == ep.rewards
    assert len(e.steps) == len(ep.steps)
    assert e.steps[0].picks == ep.steps[0].picks
    assert float(e.steps[0].logprob) == float(ep.steps[0].logprob)
    assert (e.steps[0].state.numeric == ep.steps[0].state.numeric).all()


def test_checkpoint_roundtrip_and_latest(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny", seed=1)
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    save_checkpoint(cfg, 3, p, c, opt)
    n, path = latest_checkpoint(cfg)
    assert n == 3 and path == checkpoint_path(cfg, 3)
    p2 = PolicyModel(tiny_config(tables))
    c2 = CriticModel(critic_config(tables))
    opt2 = torch.optim.Adam(list(p2.parameters()) + list(c2.parameters()), lr=1e-3)
    assert load_checkpoint(path, p2, c2, opt2) == 3
    for a, b in zip(p.state_dict().values(), p2.state_dict().values()):
        assert torch.equal(a, b)
