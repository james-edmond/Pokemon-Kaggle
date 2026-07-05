import json
import torch
from ptcg.actors import collect_round_worker, play_versus, run_actor_pool
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, load_round, round_dir, save_checkpoint
from dataclasses import asdict


def _seed_run(tmp_path, games):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                      games_per_round=games, actors=1)
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    return tables, cfg, p


def test_collect_round_worker_writes_games(tmp_path):
    tables, cfg, _ = _seed_run(tmp_path, 2)
    stats = collect_round_worker(
        (json.dumps(asdict(cfg)), 0, 0, 2, str(tmp_path / "checkpoint-0000.pt")))
    assert stats["games"] == 2 and stats["steps"] >= 10
    eps = load_round(cfg, 0)
    assert len(eps) == 2
    assert all(5 <= len(e.steps) <= 1000 for e in eps)
    # priv_viz was on: some step in some game carries opp deck entity tokens
    from ptcg.featurize import KIND_ENTITY, OWNER_OPP
    found = any(
        any((s.priv_state.zone[i] == 1 and s.priv_state.owner[i] == OWNER_OPP
             and s.priv_state.kind[i] == KIND_ENTITY)
            for i in range(s.priv_state.n))
        for e in eps for s in e.steps)
    assert found


def test_play_versus_random(tmp_path):
    tables, cfg, p = _seed_run(tmp_path, 1)
    deck = load_sample_deck()
    g = torch.Generator().manual_seed(9)
    r = play_versus(p, "random", tables, (deck, list(deck)), g, model_seat=0)
    assert r in (0, 1)
