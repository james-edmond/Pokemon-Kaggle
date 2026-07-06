from dataclasses import asdict

import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.rollout import play_game
from ptcg.trainloop import (METRIC_FIELDS, TrainConfig, _config_drift_warnings,
                            _metrics_path, append_metrics, checkpoint_path,
                            latest_checkpoint, load_checkpoint, load_round,
                            prune_checkpoints, read_metrics, round_dir,
                            save_checkpoint, save_game, truncate_metrics)


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


def test_prune_checkpoints_retention_and_idempotence(tmp_path):
    cfg = TrainConfig(run_dir=str(tmp_path), eval_every=5)
    for n in range(13):  # checkpoint-0000 .. checkpoint-0012
        checkpoint_path(cfg, n).write_bytes(b"")
    deleted = prune_checkpoints(cfg, current_round=12)
    survivors = {int(f.stem.split("-")[1])
                 for f in tmp_path.glob("checkpoint-*.pt")}
    assert survivors == {0, 5, 10, 11, 12}
    assert set(deleted) == {f"checkpoint-{n:04d}.pt"
                            for n in (1, 2, 3, 4, 6, 7, 8, 9)}
    # idempotent: a second prune deletes nothing and leaves the same survivors
    assert prune_checkpoints(cfg, current_round=12) == []
    survivors2 = {int(f.stem.split("-")[1])
                  for f in tmp_path.glob("checkpoint-*.pt")}
    assert survivors2 == {0, 5, 10, 11, 12}


def test_truncate_metrics_atomic(tmp_path):
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    for n in range(4):  # rounds 0-3
        append_metrics(cfg, dict(round=n, kind="train", games=2, steps=10))
    truncate_metrics(cfg, 2)
    rows = read_metrics(cfg)
    assert [int(r["round"]) for r in rows] == [0, 1]
    assert all(r["kind"] == "train" for r in rows)
    # header survived intact
    header = _metrics_path(cfg).read_text().splitlines()[0]
    assert header.split(",") == METRIC_FIELDS
    # atomic write left no stray tmp file
    assert not (tmp_path / "metrics.csv.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_config_drift_warnings_exact_list():
    base = TrainConfig(run_dir="x", model_size="tiny", minibatch=64,
                       eval_every=5, games_per_round=192, device="cpu")
    ck_config = asdict(base)
    # no drift and eval_every == 5 -> no warnings
    assert _config_drift_warnings(ck_config, base) == []
    # drift on every guarded key plus eval_every != 5
    drifted = TrainConfig(run_dir="x", model_size="student", minibatch=512,
                          eval_every=10, games_per_round=96, device="cuda")
    assert _config_drift_warnings(ck_config, drifted) == [
        "warning: resume config drift: model_size checkpoint=tiny cli=student",
        "warning: resume config drift: minibatch checkpoint=64 cli=512",
        "warning: resume config drift: eval_every checkpoint=5 cli=10",
        "warning: resume config drift: games_per_round checkpoint=192 cli=96",
        "warning: resume config drift: device checkpoint=cpu cli=cuda",
        "warning: eval_every=10 != 5; eval reference offsets (5/15) and "
        "pruning protection assume eval_every=5",
    ]
