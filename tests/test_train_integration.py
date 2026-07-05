from ptcg.trainloop import (TrainConfig, latest_checkpoint, read_metrics,
                            round_dir, train)


def _cfg(tmp_path):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                       games_per_round=2, actors=1, epochs=1, minibatch=64,
                       eval_every=999, device="cpu", seed=5)


def test_one_round_end_to_end_then_resume(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg, max_rounds=1)
    n, _ = latest_checkpoint(cfg)
    assert n == 1
    rows = read_metrics(cfg)
    assert len(rows) == 1 and rows[0]["kind"] == "train"
    assert int(rows[0]["round"]) == 0 and int(rows[0]["steps"]) > 5
    assert not round_dir(cfg, 0).exists()  # consumed and deleted

    train(cfg, max_rounds=2)  # resume: must NOT redo round 0
    n2, _ = latest_checkpoint(cfg)
    assert n2 == 2
    rows = read_metrics(cfg)
    assert [int(r["round"]) for r in rows if r["kind"] == "train"] == [0, 1]
