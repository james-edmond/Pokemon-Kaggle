import numpy as np
import torch

from .cards import card_row
from .featurize import F_PRIZEN, KIND_ENTITY, OWNER_OPP, OWNER_SELF
from .model import collate_states

AREA_HAND = 2
_PSUM_SELF, _PSUM_OPP = 1, 2  # fixed special-token rows


def compute_gae(values, terminal_reward, lam=0.95, gamma=1.0):
    """Per-seat GAE with terminal-only reward; matches the phase-1 smoke test."""
    adv_acc, advs = 0.0, []
    for j in reversed(range(len(values))):
        nxt = values[j + 1] if j + 1 < len(values) else terminal_reward
        delta = gamma * nxt - values[j]
        adv_acc = delta + gamma * lam * adv_acc
        advs.append(adv_acc)
    advs.reverse()
    rets = [a + v for a, v in zip(advs, values)]
    return advs, rets


def assemble_advantages(episodes, critic, device=None, lam=0.95, gamma=1.0,
                        normalize=True):
    steps = [s for ep in episodes for s in ep.steps]
    with torch.no_grad():
        chunks = []
        for lo in range(0, len(steps), 256):
            batch = collate_states([s.priv_state for s in steps[lo:lo + 256]])
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            chunks.append(critic(batch).cpu())
        pv = torch.cat(chunks) if chunks else torch.zeros(0, 2)
    adv = torch.zeros(len(steps))
    ret = torch.zeros(len(steps))
    off = 0
    for ep in episodes:
        idx = list(range(off, off + len(ep.steps)))
        off += len(ep.steps)
        for seat in (0, 1):
            rows = [i for i in idx if steps[i].player == seat]
            vals = [float(pv[i, seat]) for i in rows]
            a, r = compute_gae(vals, ep.rewards[seat], lam, gamma)
            for k, i in enumerate(rows):
                adv[i] = a[k]
                ret[i] = r[k]
    if normalize and len(steps) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    old_lp = torch.tensor([s.logprob for s in steps])
    return steps, old_lp, adv, ret


def ppo_policy_loss(new_lp, old_lp, adv, clip=0.2):
    ratio = (new_lp - old_lp).exp()
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    pg = -torch.min(ratio * adv, clipped * adv).mean()
    approx_kl = (old_lp - new_lp).mean()
    return pg, ratio, approx_kl


def aux_targets(steps, tables, opp_decks):
    """Targets for the train-only aux heads, from stored tensors only.

    prize_diff: (opp prizes remaining - own prizes remaining) read from the
    public state's player-summary rows (positive = acting seat is ahead).
    decklist: opponent decklist as card-table-row counts. `opp_decks` is
    either a single `list[int]` (phase-2 behavior: one shared deck used for
    every step) or a list of per-step `list[int]` aligned with `steps` (each
    step's own opponent deck, e.g. for multi-deck league play).
    hand: true opponent hand counts from the privileged state. The privileged
    view is seat-0-fixed, so the acting seat's opponent is OWNER_OPP when
    player==0 and OWNER_SELF when player==1.
    """
    n_rows = tables.n_rows
    B = len(steps)
    pd = torch.zeros(B)
    dl = torch.zeros(B, n_rows)
    hd = torch.zeros(B, n_rows)
    shared = bool(opp_decks) and isinstance(opp_decks[0], int)
    shared_vec = None
    if shared:
        shared_vec = torch.zeros(n_rows)
        for cid in opp_decks:
            shared_vec[card_row(cid, n_rows)] += 1.0
    for i, s in enumerate(steps):
        num = s.state.numeric
        pd[i] = float(num[_PSUM_OPP, F_PRIZEN] - num[_PSUM_SELF, F_PRIZEN]) * 6.0
        if shared:
            dl[i] = shared_vec
        else:
            for cid in opp_decks[i]:
                dl[i, card_row(cid, n_rows)] += 1.0
        opp_owner = OWNER_OPP if s.player == 0 else OWNER_SELF
        pv = s.priv_state
        rows = np.where((pv.zone[:pv.n] == AREA_HAND)
                        & (pv.owner[:pv.n] == opp_owner)
                        & (pv.kind[:pv.n] == KIND_ENTITY))[0]
        for r in rows:
            hd[i, int(pv.card[r])] += 1.0
    return pd, dl, hd
