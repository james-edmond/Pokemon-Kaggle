import json
from dataclasses import asdict
import torch
from ptcg.actors import league_round_worker
from ptcg.cards import build_tables
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.league import snapshot
from ptcg.trainloop import (TrainConfig, save_checkpoint, load_round,
                            learner_update)


def _seed_run(tmp_path, games, **kw):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                      games_per_round=games, actors=1, **kw)
    p = PolicyModel(tiny_config(tables)); c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    return tables, cfg, p


def test_league_worker_writes_mixed_games(tmp_path):
    # pool has one snapshot so non-mirror pool games occur
    tables, cfg, p = _seed_run(tmp_path, 3, mirror_frac=0.34, pool_frac=0.66,
                               random_frac=0.0, pool_cap=18)
    snapshot(cfg, 0, p)
    stats = league_round_worker(
        (json.dumps(asdict(cfg)), 1, 0, 3, str(tmp_path / "checkpoint-0000.pt")))
    assert stats["games"] == 3 and stats["steps"] > 10
    eps = load_round(cfg, 1)
    assert len(eps) == 3
    # every episode records its decks and collected seats
    for e in eps:
        assert e.decks[0] and e.decks[1]
        assert set(e.collected_seats) <= {0, 1}
        assert all(s.player in e.collected_seats for s in e.steps)


def test_learner_update_consumes_multideck_episodes(tmp_path):
    tables, cfg, p = _seed_run(tmp_path, 2, mirror_frac=1.0, pool_frac=0.0,
                               random_frac=0.0, epochs=1, minibatch=64)
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=3e-4)
    league_round_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                         str(tmp_path / "checkpoint-0000.pt")))
    eps = load_round(cfg, 0)
    m = learner_update(p, c, opt, eps, cfg, tables, opp_deck=None)
    assert m["ratio_drift"] < 1e-6 and m["steps"] > 10
    for k in ("loss_pg", "loss_aux", "entropy"):
        assert torch.isfinite(torch.tensor(m[k])), k
