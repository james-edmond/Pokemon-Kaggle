import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.rollout import play_game
from ptcg.trainloop import TrainConfig, learner_update


def test_learner_update_ratio_gate_and_metrics():
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=3e-4)
    g = torch.Generator().manual_seed(6)
    eps = [play_game(policy, (deck, list(deck)), tables, generator=g,
                     priv_viz=True) for _ in range(2)]
    cfg = TrainConfig(run_dir="unused", model_size="tiny", epochs=2,
                      minibatch=64, device="cpu")
    before = [p.clone() for p in policy.parameters()]
    m = learner_update(policy, critic, optim, eps, cfg, tables, deck)
    # same weights collected the data: drift must be ~0 (same device/config)
    assert m["ratio_drift"] < 1e-6
    assert m["epochs_ran"] >= 1 and m["steps"] > 10
    for k in ("loss_pg", "loss_v", "loss_critic", "loss_aux",
              "entropy", "approx_kl"):
        assert torch.isfinite(torch.tensor(m[k])), k
    assert any(not torch.equal(a, b)
               for a, b in zip(before, policy.parameters()))


def test_learner_update_aborts_on_stale_policy():
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(7)
    eps = [play_game(policy, (deck, list(deck)), tables, generator=g,
                     priv_viz=True)]
    other = PolicyModel(tiny_config(tables))  # different weights entirely
    cfg = TrainConfig(run_dir="unused", model_size="tiny", device="cpu")
    import pytest
    with pytest.raises(RuntimeError):
        learner_update(other, critic, optim, eps, cfg, tables, deck)
