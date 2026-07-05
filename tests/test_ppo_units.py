import math

import torch
from ptcg.ppo import compute_gae, ppo_policy_loss


def test_gae_hand_computed():
    # values [0.5, 0.0], terminal reward 1.0, lam=0.5, gamma=1.0
    # delta1 = 0.0 - 0.5 = -0.5 ; delta2 = 1.0 - 0.0 = 1.0
    # adv2 = 1.0 ; adv1 = -0.5 + 0.5*1.0 = 0.0
    adv, ret = compute_gae([0.5, 0.0], 1.0, lam=0.5, gamma=1.0)
    assert math.isclose(adv[0], 0.0, abs_tol=1e-9)
    assert math.isclose(adv[1], 1.0, abs_tol=1e-9)
    assert math.isclose(ret[0], 0.5, abs_tol=1e-9)
    assert math.isclose(ret[1], 1.0, abs_tol=1e-9)


def test_gae_matches_smoke_test_reference():
    # the phase-1 smoke test's _gae over deltas must agree
    def _gae(deltas, gamma=1.0, lam=0.95):
        adv, out = 0.0, []
        for d in reversed(deltas):
            adv = d + gamma * lam * adv
            out.append(adv)
        return list(reversed(out))

    vals, rw = [0.2, -0.1, 0.4], -1.0
    deltas = [(vals[j + 1] if j + 1 < len(vals) else rw) - vals[j]
              for j in range(len(vals))]
    ref = _gae(deltas)
    adv, _ = compute_gae(vals, rw)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(adv, ref))


def test_ppo_policy_loss_clip_and_kl():
    old = torch.tensor([0.0, 0.0])
    new = torch.tensor([math.log(2.0), math.log(0.25)])
    adv = torch.tensor([1.0, 1.0])
    pg, ratio, kl = ppo_policy_loss(new, old, adv, clip=0.2)
    # ratios 2.0 and 0.25; positive adv -> min(r, clip(r)) = (1.2, 0.25)
    assert torch.allclose(ratio, torch.tensor([2.0, 0.25]))
    assert math.isclose(float(pg), -(1.2 + 0.25) / 2, abs_tol=1e-6)
    assert math.isclose(float(kl), float((old - new).mean()), abs_tol=1e-9)
