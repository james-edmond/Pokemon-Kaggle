from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .model import collate_selects, collate_states


@dataclass
class SelectDecision:
    picks: list
    logprob: float
    entropy: float


def run_pick_loop(model, trunk, state_batch, sel_batch, *, forced=None, generator=None):
    O = sel_batch["opt_type"].shape[1]
    max_count = int(sel_batch["max_count_t"][0])
    picked = torch.zeros((1, O + 1), dtype=torch.bool)
    picks, logp, ent = [], trunk.new_zeros(()), trunk.new_zeros(())
    step = 0
    while True:
        logits = model.option_logits(trunk, state_batch, sel_batch, picked)
        dist = Categorical(logits=logits)
        if forced is None:
            a = dist.sample() if generator is None else torch.multinomial(
                dist.probs.squeeze(0), 1, generator=generator)
            a = a.reshape(1)
        else:
            a = torch.tensor([forced[step] if step < len(forced) else O])
        logp = logp + dist.log_prob(a).squeeze(0)
        ent = ent + dist.entropy().squeeze(0)
        if int(a) == O:  # done
            break
        picks.append(int(a))
        picked[0, int(a)] = True
        step += 1
        if len(picks) == max_count:
            break
    return picks, logp, ent


def sample_select(model, ts, es, generator=None) -> SelectDecision:
    sb = collate_states([ts])
    selb = collate_selects([es])
    with torch.no_grad():
        trunk = model.encode(sb)
        picks, logp, ent = run_pick_loop(model, trunk, sb, selb, generator=generator)
    return SelectDecision(picks, float(logp), float(ent))


def replay_logprob(model, states, sels, picks_list):
    out = []
    for ts, es, picks in zip(states, sels, picks_list):
        sb = collate_states([ts])
        selb = collate_selects([es])
        trunk = model.encode(sb)
        _, logp, _ = run_pick_loop(model, trunk, sb, selb, forced=list(picks))
        out.append(logp)
    return torch.stack(out)
