import json
from dataclasses import asdict
import torch
from ptcg.actors import league_eval_worker
from ptcg.cards import build_tables
from ptcg.decks import SAMPLE
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, save_checkpoint


def test_league_eval_worker_per_deck(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    p = PolicyModel(tiny_config(tables)); c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()))
    save_checkpoint(cfg, 0, p, c, opt)
    out = league_eval_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                              str(tmp_path / "checkpoint-0000.pt"),
                              "random", SAMPLE))
    assert out["games"] == 2 and 0 <= out["wins"] <= 2
