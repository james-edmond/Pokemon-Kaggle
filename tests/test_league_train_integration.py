from ptcg.trainloop import TrainConfig, latest_checkpoint, read_metrics, train
from ptcg.league import snapshot_rounds


def _cfg(tmp_path):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                       games_per_round=2, actors=1, epochs=1, minibatch=64,
                       eval_every=999, snapshot_every=1, pool_cap=4,
                       mirror_frac=0.5, pool_frac=0.5, random_frac=0.0,
                       device="cpu", seed=5)


def test_league_train_snapshots_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg, max_rounds=2)
    assert latest_checkpoint(cfg)[0] == 2
    # snapshot_every=1 -> pool has snapshots for rounds 1 and 2
    assert snapshot_rounds(cfg) == [1, 2]
    rows = [r for r in read_metrics(cfg) if r["kind"] == "train"]
    assert [int(r["round"]) for r in rows] == [0, 1]
    train(cfg, max_rounds=3)               # resume: no redo, pool grows
    assert latest_checkpoint(cfg)[0] == 3
    assert 3 in snapshot_rounds(cfg)
