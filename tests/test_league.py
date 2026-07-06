import random
import torch
from ptcg.cards import build_tables
from ptcg.league import (league_dir, snapshot, snapshot_rounds, prune_pool,
                        sample_opponent, load_opponent)
from ptcg.model import PolicyModel, tiny_config
from ptcg.trainloop import TrainConfig


def _cfg(tmp_path, **kw):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny", **kw)


def test_snapshot_and_prune(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path)
    p = PolicyModel(tiny_config(tables))
    for r in (0, 5, 10, 15, 20, 25):
        snapshot(cfg, r, p)
    assert snapshot_rounds(cfg) == [0, 5, 10, 15, 20, 25]
    kept = prune_pool(cfg, cap=4, anchors=2)
    # 2 earliest anchors (0,5) + newest up to cap: total 4 -> {0,5,20,25}
    assert kept == [0, 5, 20, 25]
    assert snapshot_rounds(cfg) == [0, 5, 20, 25]


def test_snapshot_roundtrip_loads_same_weights(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path)
    p = PolicyModel(tiny_config(tables))
    path = snapshot(cfg, 3, p)
    q = load_opponent(str(path), tables, cfg)
    for a, b in zip(p.state_dict().values(), q.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())


def test_sample_opponent_mixture(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path, mirror_frac=0.0, pool_frac=1.0, random_frac=0.0)
    p = PolicyModel(tiny_config(tables))
    snapshot(cfg, 0, p)
    kinds = [sample_opponent(cfg, 5, random.Random(i))[0] for i in range(10)]
    assert set(kinds) == {"pool"}          # pool_frac=1 always draws pool
    cfg2 = _cfg(tmp_path, mirror_frac=1.0, pool_frac=0.0, random_frac=0.0)
    assert sample_opponent(cfg2, 5, random.Random(0)) == ("current", None)
    # empty pool falls back to current even with pool_frac=1
    cfg3 = _cfg(str(tmp_path) + "_empty", mirror_frac=0.0, pool_frac=1.0,
                random_frac=0.0)
    assert sample_opponent(cfg3, 5, random.Random(0))[0] == "current"
