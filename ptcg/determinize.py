"""Sample engine-ready hidden-info assignments (determinizations).

Opponent side merges three information layers:
  1. visible cards (discard, board incl. attachments/pre-evolutions,
     revealed prizes, their stadium) are fixed and excluded;
  2. BeliefTracker constraints: known hand/deck membership are hard
     minimums, the hidden pool must land in hand+deck;
  3. aux heads: dl_lam (expected full-decklist counts per card row)
     weights unknown-card sampling, hd_lam weights which unknowns sit in
     the hand. A 4-per-card-id cap (basic energy exempt) and a 1-ACE-SPEC
     cap keep samples realistic (approximates the 4-per-name rule; the
     inference tables carry no names).
Own side: unseen = decklist - hand - visible; the deck/prize split is
sampled uniformly (which cards are prized IS uniform; ordering is
engine-shuffled regardless of what we supply).
Contract: never raises; exact zone counts whenever the obs is well-formed.
"""
import random
from collections import Counter
from dataclasses import dataclass

from .cards import (A_ACESPEC, A_BASIC, A_IS_BASIC_ENERGY, A_IS_POKEMON,
                    N_RESERVED)

_BASIC_ENERGY_IDS = (1, 2, 3, 4, 5, 6, 7, 8)   # card id == EnergyType 1..8


@dataclass(frozen=True)
class Determinization:
    your_deck: list
    your_prize: list
    opp_deck: list
    opp_prize: list
    opp_hand: list
    opp_active: list
    opp_decklist: list      # full 60 for featurizing opponent nodes


def _board_ids(pstate):
    out = []
    for pk in (pstate.get("active") or []) + (pstate.get("bench") or []):
        if not pk:
            continue
        out.append(pk["id"])
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in pk.get(grp) or []:
                if c:
                    out.append(c["id"])
    return out


def _visible_counts(obs, pidx):
    p = obs["current"]["players"][pidx]
    ids = [c["id"] for c in p.get("discard") or [] if c]
    ids += _board_ids(p)
    ids += [c["id"] for c in p.get("prize") or [] if c]     # revealed prizes
    for c in obs["current"].get("stadium") or []:
        if c and c.get("playerIndex") == pidx:
            ids.append(c["id"])
    return Counter(ids)


def _energy_filler(obs, opp):
    """A basic-energy id matching the opponent's dominant board energy."""
    try:
        p = obs["current"]["players"][opp]
        et = Counter()
        for pk in (p.get("active") or []) + (p.get("bench") or []):
            if pk:
                for e in pk.get("energies") or []:
                    et[int(e)] += 1
        for e, _ in et.most_common():
            if 1 <= e <= 8:
                return e
    except Exception:
        pass
    return 3                                                # water energy


def _fallback_basic(tables):
    """Lowest-id Basic Pokemon in the pool (a valid, safe active guess)."""
    attr = tables.attr
    for row in range(N_RESERVED, tables.n_rows):
        if attr[row, A_IS_POKEMON] and attr[row, A_BASIC]:
            return row - N_RESERVED
    return _BASIC_ENERGY_IDS[2]     # unreachable with real tables


def _sample_counts(weights, k, rng, assigned, attr, n_rows):
    """k card ids by weighted draw; count-decrement without replacement.

    weights: dict row -> float. assigned: Counter id -> already-placed count
    (visible + known + sampled so far), used for the 4-per-id cap (basic
    energy exempt) and the 1-ACE-SPEC cap.
    """
    out = []
    w = {r: float(v) for r, v in weights.items() if v > 0}
    ace_used = any(attr[cid + N_RESERVED, A_ACESPEC]
                   for cid in assigned if 0 <= cid + N_RESERVED < n_rows)
    for _ in range(k):
        live = []
        tot = 0.0
        for r, v in w.items():
            if v <= 0:
                continue
            cid = r - N_RESERVED
            if not attr[r, A_IS_BASIC_ENERGY] and assigned[cid] >= 4:
                continue
            if ace_used and attr[r, A_ACESPEC]:
                continue
            live.append((r, v))
            tot += v
        if tot <= 0 or not live:
            break
        x = rng.random() * tot
        acc = 0.0
        for r, v in live:
            acc += v
            if acc >= x:
                cid = r - N_RESERVED
                out.append(cid)
                assigned[cid] += 1
                w[r] = v - 1.0
                if attr[r, A_ACESPEC]:
                    ace_used = True
                break
    return out


def _pick_indices(weights, k, rng):
    """k distinct indices, probability proportional to weight (removal)."""
    w = {i: float(v) for i, v in enumerate(weights) if v > 0}
    out = []
    for _ in range(min(k, len(w))):
        tot = sum(w.values())
        if tot <= 0:
            out.extend(list(w.keys())[:k - len(out)])
            break
        x = rng.random() * tot
        acc = 0.0
        for i, v in list(w.items()):
            acc += v
            if acc >= x:
                out.append(i)
                del w[i]
                break
    return out


def sample_determinization(obs, me, my_deck, belief, dl_lam, hd_lam,
                           tables, rng):
    try:
        return _sample(obs, me, my_deck, belief, dl_lam, hd_lam, tables, rng)
    except Exception:
        return filler_determinization(obs, me, my_deck, tables, rng)


def filler_determinization(obs, me, my_deck, tables, rng):
    """Constraint-free fallback: no belief, no aux guidance."""
    try:
        return _sample(obs, me, my_deck, None, None, None, tables, rng)
    except Exception:
        return Determinization([], [], [], [], [], [], [])


def _sample(obs, me, my_deck, belief, dl_lam, hd_lam, tables, rng):
    cur = obs["current"]
    opp = 1 - me
    pme, popp = cur["players"][me], cur["players"][opp]
    attr, n_rows = tables.attr, tables.n_rows
    filler = _energy_filler(obs, opp)

    # ---------- my side ----------
    unseen = Counter(int(c) for c in my_deck)
    unseen.subtract(Counter(c["id"] for c in pme.get("hand") or [] if c))
    unseen.subtract(_visible_counts(obs, me))
    for c in cur.get("looking") or []:
        if c and c.get("playerIndex") == me:
            unseen[c["id"]] -= 1
    pool = [cid for cid, n in unseen.items() for _ in range(max(n, 0))]
    rng.shuffle(pool)
    n_deck = int(pme["deckCount"])
    my_prize = list(pme.get("prize") or [])
    n_unrev = sum(1 for c in my_prize if c is None)
    while len(pool) < n_deck + n_unrev:
        pool.append(int(rng.choice(my_deck)))
    your_deck = pool[:n_deck]
    it = iter(pool[n_deck:n_deck + n_unrev])
    your_prize = [(c["id"] if c else next(it)) for c in my_prize]

    # ---------- opponent side ----------
    vis = _visible_counts(obs, opp)
    kh = Counter(dict(belief.opp_hand)) if belief else Counter()
    kd = Counter(dict(belief.opp_deck)) if belief else Counter()
    kp = Counter(dict(belief.opp_hidden_pool)) if belief else Counter()
    n_hand = int(popp["handCount"])
    n_deck_o = int(popp["deckCount"])
    opp_prize_l = list(popp.get("prize") or [])
    n_unrev_o = sum(1 for c in opp_prize_l if c is None)
    hidden_total = n_hand + n_deck_o + n_unrev_o
    known = [cid for cnt in (kh, kd, kp) for cid, n in cnt.items()
             for _ in range(n)]
    n_unknown = max(0, hidden_total - len(known))

    assigned = Counter(vis)
    for cid in known:
        assigned[cid] += 1
    weights = {}
    for row in range(N_RESERVED, n_rows):
        if not attr[row].any():
            continue
        base = float(dl_lam[row]) if dl_lam is not None else 0.0
        w = base - assigned[row - N_RESERVED]   # predicted total minus placed
        if w > 0:
            weights[row] = w
    unknown = _sample_counts(weights, n_unknown, rng, assigned, attr, n_rows)
    while len(unknown) < n_unknown:
        unknown.append(filler)

    # zone assignment: known hand first, hd_lam weights the rest of the hand
    hand = [cid for cid, n in kh.items() for _ in range(n)][:n_hand]
    pool_items = [cid for cid, n in kp.items() for _ in range(n)]
    cand = pool_items + unknown              # order marks provenance
    need = n_hand - len(hand)
    if need > 0:
        hw = [(float(hd_lam[c + N_RESERVED]) if hd_lam is not None
               and 0 <= c + N_RESERVED < n_rows else 0.0) + 0.05
              for c in cand]
        idx = set(_pick_indices(hw, need, rng))
        hand += [cand[i] for i in sorted(idx)]
        rem_pool = [cand[i] for i in range(len(pool_items)) if i not in idx]
        rem_unknown = [cand[i] for i in range(len(pool_items), len(cand))
                       if i not in idx]
    else:
        rem_pool, rem_unknown = pool_items, list(unknown)
    while len(hand) < n_hand:
        hand.append(filler)
    hand = hand[:n_hand]

    # deck: known-deck minimums, then ALL leftover pool cards (pool means
    # hand-or-deck membership - never prized), then unknowns; prizes draw
    # only from unknowns. Pool overflow past deck capacity (impossible
    # unless the tracker overcounts) spills to the prize pool defensively.
    known_deck_list = [cid for cid, n in kd.items() for _ in range(n)]
    rng.shuffle(rem_unknown)
    deck = known_deck_list + rem_pool
    space = n_deck_o - len(deck)
    deck += rem_unknown[:max(0, space)]
    rest = rem_unknown[max(0, space):]
    if len(deck) > n_deck_o:
        rest = deck[n_deck_o:] + rest
        deck = deck[:n_deck_o]
    while len(deck) < n_deck_o:
        deck.append(filler)

    prize_pool = rest
    while len(prize_pool) < n_unrev_o:
        prize_pool.append(filler)
    it2 = iter(prize_pool)
    opp_prize = [(c["id"] if c else next(it2)) for c in opp_prize_l]

    # facedown active + setup basic requirements
    opp_active = []
    oa = popp.get("active") or []
    if oa and oa[0] is None:
        best_row, best_w = None, -1.0
        for row, w in weights.items():
            if attr[row, A_IS_POKEMON] and attr[row, A_BASIC] and w > best_w:
                best_row, best_w = row, w
        opp_active = [best_row - N_RESERVED if best_row is not None
                      else _fallback_basic(tables)]
    if int(cur.get("turn") or 0) == 0:
        has_basic = any(
            0 <= c + N_RESERVED < n_rows
            and attr[c + N_RESERVED, A_IS_POKEMON]
            and attr[c + N_RESERVED, A_BASIC] for c in deck)
        if not has_basic and deck:
            deck[-1] = _fallback_basic(tables)

    filled_prizes = [x for slot, x in zip(opp_prize_l, opp_prize)
                     if slot is None]
    decklist = ([c for c in vis.elements()] + hand + deck + filled_prizes)
    while len(decklist) < 60:
        decklist.append(filler)
    decklist = decklist[:60]

    return Determinization(
        [int(x) for x in your_deck], [int(x) for x in your_prize],
        [int(x) for x in deck], [int(x) for x in opp_prize],
        [int(x) for x in hand], [int(x) for x in opp_active],
        [int(x) for x in decklist])
