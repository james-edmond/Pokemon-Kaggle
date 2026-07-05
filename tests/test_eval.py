import json
from dataclasses import asdict

import torch
from ptcg.actors import eval_worker
from ptcg.cards import build_tables
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, save_checkpoint


def test_eval_worker_vs_random(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()))
    save_checkpoint(cfg, 0, p, c, opt)
    out = eval_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                       str(tmp_path / "checkpoint-0000.pt"), "random"))
    assert out["games"] == 2 and 0 <= out["wins"] <= 2
