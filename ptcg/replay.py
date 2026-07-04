import torch
from torch.distributions import Categorical

from .model import collate_selects, collate_states


def batched_replay(model, trunk, sb, selb, picks_list):
    """Deterministic lockstep replay of stored pick sequences over a shared trunk.

    Mirrors action.run_pick_loop(forced=...) semantics per row: force each
    stored pick in order; force one done step only where the stored sequence
    ended before max_count; never re-sample. picked is cloned before each
    mutation (same autograd hazard as the B==1 loop).
    """
    B, O = selb["opt_type"].shape
    dev = trunk.device
    max_count = selb["max_count_t"].to(dev)
    n_picks = torch.tensor([len(p) for p in picks_list], device=dev)
    depth = int(n_picks.max().item()) if B else 0
    picked = torch.zeros((B, O + 1), dtype=torch.bool, device=dev)
    logp = trunk.new_zeros(B)
    ent = trunk.new_zeros(B)
    for step in range(depth + 1):
        takes_pick = n_picks > step
        forces_done = (n_picks == step) & (n_picks < max_count)
        alive = takes_pick | forces_done
        if not bool(alive.any()):
            break
        actions = torch.full((B,), O, dtype=torch.int64, device=dev)
        for i, picks in enumerate(picks_list):
            if step < len(picks):
                actions[i] = picks[step]
        logits = model.option_logits(trunk, sb, selb, picked)
        dist = Categorical(logits=logits)
        lp = dist.log_prob(actions)
        en = dist.entropy()
        zero = torch.zeros((), device=dev, dtype=lp.dtype)
        logp = logp + torch.where(alive, lp, zero)
        ent = ent + torch.where(alive, en, zero)
        new_picked = picked.clone()
        rows = takes_pick.nonzero(as_tuple=True)[0]
        new_picked[rows, actions[rows]] = True
        picked = new_picked
    return logp, ent


def replay_logprob_batched(model, states, sels, picks_list, device=None):
    sb = collate_states(states)
    selb = collate_selects(sels)
    if device is not None:
        sb = {k: v.to(device) for k, v in sb.items()}
        selb = {k: v.to(device) for k, v in selb.items()}
    trunk = model.encode(sb)
    return batched_replay(model, trunk, sb, selb, picks_list)
