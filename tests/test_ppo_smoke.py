import torch
from ptcg.action import replay_logprob
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import (CriticModel, PolicyModel, collate_states,
                        critic_config, tiny_config)
from ptcg.rollout import play_game


def _gae(deltas, gamma=1.0, lam=0.95):
    adv, out = 0.0, []
    for d in reversed(deltas):
        adv = d + gamma * lam * adv
        out.append(adv)
    return list(reversed(out))


def test_ratio_is_one_and_gradients_flow():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    g = torch.Generator().manual_seed(2)
    eps = [play_game(m, (deck, list(deck)), tables, generator=g) for _ in range(2)]

    steps = [s for ep in eps for s in ep.steps]
    old_lp = torch.tensor([s.logprob for s in steps])
    new_lp = replay_logprob(m, [s.state for s in steps],
                            [s.esel for s in steps], [s.picks for s in steps])
    ratio = (new_lp - old_lp).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), \
        "shared pick loop must reproduce actor log-probs exactly"

    # per-seat advantages from the privileged critic, gamma=1 terminal reward
    with torch.no_grad():
        pv = critic(collate_states([s.priv_state for s in steps]))
    adv, ret = [], []
    off = 0
    for ep in eps:
        idx = list(range(off, off + len(ep.steps)))
        off += len(ep.steps)
        for seat in (0, 1):
            rows = [i for i in idx if steps[i].player == seat]
            vals = [float(pv[i, seat]) for i in rows]
            rw = ep.rewards[seat]
            deltas = [
                (vals[j + 1] if j + 1 < len(vals) else rw) - vals[j]
                for j in range(len(vals))
            ]
            a = _gae(deltas)
            adv += [(rows[j], a[j]) for j in range(len(rows))]
            ret += [(rows[j], a[j] + vals[j]) for j in range(len(rows))]
    order = [i for i, _ in sorted(adv)]
    advt = torch.tensor([a for _, a in sorted(adv)])
    rett = torch.tensor([r for _, r in sorted(ret)])

    opt = torch.optim.Adam(list(m.parameters()) + list(critic.parameters()), lr=3e-4)
    losses = []
    for _ in range(3):
        new_lp = replay_logprob(m, [steps[i].state for i in order],
                                [steps[i].esel for i in order],
                                [steps[i].picks for i in order])
        ratio = (new_lp - old_lp[order]).exp()
        clipped = torch.clamp(ratio, 0.8, 1.2)
        pg = -torch.min(ratio * advt, clipped * advt).mean()
        trunkb = collate_states([steps[i].state for i in order])
        v = m.public_value(m.encode(trunkb))
        vloss = ((v - rett) ** 2).mean()
        pvb = critic(collate_states([steps[i].priv_state for i in order]))
        closs = ((pvb[range(len(order)), [steps[i].player for i in order]] - rett) ** 2).mean()
        loss = pg + 0.5 * vloss + 0.5 * closs
        opt.zero_grad()
        loss.backward()
        for name, p in list(m.named_parameters()) + list(critic.named_parameters()):
            assert p.grad is None or torch.isfinite(p.grad).all(), name
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], "loss should decrease when overfitting one batch"
