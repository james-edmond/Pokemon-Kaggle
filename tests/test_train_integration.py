import pytest
import torch

from ptcg.trainloop import (TrainConfig, checkpoint_path, latest_checkpoint,
                            load_checkpoint, model_config_for, read_metrics,
                            round_dir, save_checkpoint, train)


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


def _build_on(cfg, tables):
    """Construct policy/critic/optim the way train() does: params on cfg.device
    BEFORE the optimizer is created."""
    from ptcg.model import CriticModel, PolicyModel, critic_config
    policy = PolicyModel(model_config_for(cfg.model_size, tables)).to(cfg.device)
    critic = CriticModel(critic_config(tables)).to(cfg.device)
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=cfg.lr)
    return policy, critic, optim


def _grad_step(policy, critic, optim):
    """One real backward+step. A parameter-norm surrogate loss populates a grad
    for every param on the params' device — enough to exercise Adam's per-param
    state, which is where the resume-device bug lives (independent of the actual
    PPO loss)."""
    loss = (sum((p ** 2).sum() for p in policy.parameters())
            + sum((p ** 2).sum() for p in critic.parameters()))
    optim.zero_grad()
    loss.backward()
    optim.step()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="resume-device regression is only meaningful on cuda")
def test_cuda_resume_optim_state_lands_on_device(tmp_path):
    """Regression for the resume-path device bug: on a cuda resume, the Adam
    state restored from a post-update checkpoint must end up on cuda so that
    optim.step() does not hit a cuda-vs-cpu device mismatch.

    Reproduces the EXACT failing order — restore optimizer state, THEN step on
    cuda — using the real save_checkpoint/load_checkpoint round trip."""
    from ptcg.cards import build_tables
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny", device="cuda",
                      lr=3e-4, seed=5)

    # Round-0 update on cuda populates Adam state (exp_avg/exp_avg_sq on cuda);
    # save it exactly like train() saves checkpoint-0001.
    policy, critic, optim = _build_on(cfg, tables)
    _grad_step(policy, critic, optim)
    save_checkpoint(cfg, 1, policy, critic, optim)

    # Negative control: the OLD buggy order (load into CPU params, move to cuda
    # afterwards) leaves Adam state on CPU and crashes on step — proves this
    # test actually exercises the reported failure.
    from ptcg.model import CriticModel, PolicyModel, critic_config
    bad_p = PolicyModel(model_config_for(cfg.model_size, tables))
    bad_c = CriticModel(critic_config(tables))
    bad_o = torch.optim.Adam(
        list(bad_p.parameters()) + list(bad_c.parameters()), lr=cfg.lr)
    load_checkpoint(checkpoint_path(cfg, 1), bad_p, bad_c, bad_o)  # params CPU
    bad_p.to("cuda"); bad_c.to("cuda")  # params move, Adam state does not
    with pytest.raises(RuntimeError, match="two devices"):
        _grad_step(bad_p, bad_c, bad_o)

    # Fixed order (what train() now does): params on cuda BEFORE load_checkpoint,
    # so optim.load_state_dict casts the CPU-loaded Adam state onto cuda.
    policy2, critic2, optim2 = _build_on(cfg, tables)
    start = load_checkpoint(checkpoint_path(cfg, 1), policy2, critic2, optim2)
    assert start == 1
    _grad_step(policy2, critic2, optim2)  # must NOT raise

    sampled = next(st["exp_avg"] for st in optim2.state.values()
                   if "exp_avg" in st)
    assert sampled.is_cuda
