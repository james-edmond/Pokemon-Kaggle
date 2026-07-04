import torch
from ptcg.action import replay_logprob, run_pick_loop
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, collate_selects, collate_states, tiny_config
from ptcg.replay import replay_logprob_batched
from ptcg.rollout import play_game


def _collect_steps(n=24):
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(3)
    steps = []
    while len(steps) < n:
        ep = play_game(m, (deck, list(deck)), tables, generator=g)
        steps.extend(ep.steps)
    return tables, m, steps[:n]


def test_batched_matches_b1_exactly():
    tables, m, steps = _collect_steps()
    states = [s.state for s in steps]
    sels = [s.esel for s in steps]
    picks = [s.picks for s in steps]
    lp1 = replay_logprob(m, states, sels, picks)
    lpb, entb = replay_logprob_batched(m, states, sels, picks)
    # Decision rule (task-1 brief): batched vs single-row CPU GEMM kernels can
    # differ in the last float bits. Measured max abs diff here is ~2.4e-7,
    # well under the 1e-6 tolerance and with no structural mismatch, so this
    # one assertion is relaxed to atol=1e-6 per the brief's rule.
    assert torch.allclose(lpb, lp1, atol=1e-6, rtol=0), (lpb - lp1).abs().max()
    # entropy reference via the B==1 forced loop
    ents = []
    for s in steps:
        sb = collate_states([s.state])
        selb = collate_selects([s.esel])
        trunk = m.encode(sb)
        _, _, ent = run_pick_loop(m, trunk, sb, selb, forced=list(s.picks))
        ents.append(ent)
    assert torch.allclose(entb, torch.stack(ents), atol=1e-6, rtol=0)


def test_batched_replay_backward():
    tables, m, steps = _collect_steps(8)
    lpb, entb = replay_logprob_batched(
        m, [s.state for s in steps], [s.esel for s in steps],
        [s.picks for s in steps])
    (lpb.sum() + entb.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in m.parameters())
