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
    # differ in the last float bits. Measured max abs diffs with the corrected
    # tgt padding mask: logp 2.384e-7, entropy 2.384e-7 — under the 1e-6
    # tolerance with no structural mismatch, so these assertions are relaxed
    # to atol=1e-6 per the brief's rule.
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


def test_option_logits_query_and_picked_sensitivity():
    # Regression for the tgt_key_padding_mask polarity bug: True means
    # IGNORE as a self-attention key, so flagging the q/done columns True
    # silently severed query and picked conditioning (the deltas below were
    # exactly 0.0) while every behavioral test stayed green. These
    # assertions pin the conditioning paths open.
    import copy
    tables, m, steps = _collect_steps()
    step = next(s for s in steps if len(s.esel.opt_type) >= 2)
    sb = collate_states([step.state])
    selb = collate_selects([step.esel])
    O = selb["opt_type"].shape[1]
    picked = torch.zeros((1, O + 1), dtype=torch.bool)
    with torch.no_grad():
        trunk = m.encode(sb)
        base = m.option_logits(trunk, sb, selb, picked)
        # (a) query conditioning: perturbing q_scalar before collation must
        # move the option logits
        es2 = copy.deepcopy(step.esel)
        es2.q_scalar = es2.q_scalar + 1.0
        pert = m.option_logits(trunk, sb, collate_selects([es2]), picked)
        fin = torch.isfinite(base) & torch.isfinite(pert)
        assert (pert[fin] - base[fin]).abs().max() > 0
        # (b) picked conditioning: setting one picked bit must move at least
        # one OTHER column's logit (the picked column itself just goes -inf)
        picked2 = picked.clone()
        picked2[0, 0] = True
        after = m.option_logits(trunk, sb, selb, picked2)
        f2 = torch.isfinite(base[0, 1:O]) & torch.isfinite(after[0, 1:O])
        assert f2.any()
        assert (after[0, 1:O][f2] - base[0, 1:O][f2]).abs().max() > 0


def test_batched_replay_backward():
    tables, m, steps = _collect_steps(8)
    lpb, entb = replay_logprob_batched(
        m, [s.state for s in steps], [s.esel for s in steps],
        [s.picks for s in steps])
    (lpb.sum() + entb.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in m.parameters())
