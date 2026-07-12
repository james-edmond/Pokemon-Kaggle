"""K-tree PUCT search over the engine's search API.

One PUCT tree per determinization; root actions/priors are shared (the
root select is the real one in every tree). Values are stored from each
node's acting seat and negamax-flipped on backup; a leaf's public_value is
only ever trusted at a MY-seat node -- the value head has a systematic
turn-phase level shift (turn-start reads |v|~0.9 for comparable advantage
vs ~0.6 mid-turn), so comparing values across phases biased the search
toward passive, turn-ending lines. Opponent nodes are still expanded (for
their actions/priors, with the same acting-seat information structure the
net trained on: that seat's belief tracker + decklist) but their value is
discarded and the descent continues through them, so every backed-up
value is my-phase or an exact terminal. Chance (draws/shuffles/coins)
stays implicit: each expansion samples one outcome and caches it
(determinized-UCT bias, mitigated by the K independent trees). The root
decision is the argmax of summed visit counts across trees over exact
pick tuples (order-preserving, safe for order-sensitive contexts, at
worst splitting equivalent votes).
"""
import math
import time
from collections import Counter
from dataclasses import dataclass

from .tracker import BeliefTracker


@dataclass
class SearchConfig:
    k_trees: int = 6
    sims_per_tree: int = 64
    c_puct: float = 1.5
    m_multipick: int = 8


@dataclass
class MoveStats:
    searched: bool = False
    trees: int = 0
    sims: int = 0
    elapsed: float = 0.0
    reason: str = ""
    root_actions: list = None   # candidate pick-tuples (populated iff searched)
    root_visits: list = None    # summed cross-tree visits, aligned with root_actions


class _Node:
    __slots__ = ("sid", "obs", "seat", "trk", "term_v",
                 "actions", "P", "N", "W", "children")

    def __init__(self, sid, obs, seat, trk, term_v=None):
        self.sid = sid
        self.obs = obs
        self.seat = seat
        self.trk = trk              # (tracker seat 0, tracker seat 1)
        self.term_v = term_v        # value for the ROOT player if terminal
        self.actions = None         # None until evaluated (leaf)
        self.P = self.N = self.W = None
        self.children = {}


def _clone(t):
    n = BeliefTracker(t.me)
    n._hand = Counter(t._hand)
    n._deck = Counter(t._deck)
    n._pool = Counter(t._pool)
    return n


def _child_trackers(pair, child_obs):
    s = child_obs["current"]["yourIndex"]
    t = _clone(pair[s])
    t.update(child_obs.get("logs") or [])
    out = list(pair)
    out[s] = t
    return tuple(out)


def _greedy_picks(model, trunk, sb, selb):
    import torch
    O = selb["opt_type"].shape[1]
    max_count = int(selb["max_count_t"][0])
    picked = torch.zeros((1, O + 1), dtype=torch.bool)
    picks = []
    while True:
        logits = model.option_logits(trunk, sb, selb, picked)
        a = int(logits.argmax(dim=-1))
        if a == O:
            break
        picks.append(a)
        picked = picked.clone()
        picked[0, a] = True
        if len(picks) == max_count:
            break
    return picks


def propose_actions(model, ts, es, select, gen, m_multipick=8):
    """Candidate pick-lists + priors for one select.

    Returns (actions: list[tuple[int,...]], priors: list[float], trunk).
    Single-pick selects (98% of nodes) cost one option_logits pass; the
    empty decline () is a candidate iff minCount==0 (prior = done column).
    Multi-pick selects use m sampled pick-lists plus the greedy one.
    """
    import torch

    from .action import run_pick_loop
    from .model import collate_selects, collate_states

    sb = collate_states([ts])
    selb = collate_selects([es])
    n = len(select["option"])
    lo, hi = int(select["minCount"]), int(select["maxCount"])
    with torch.no_grad():
        trunk = model.encode(sb)
        if hi == 1:
            O = selb["opt_type"].shape[1]
            picked = torch.zeros((1, O + 1), dtype=torch.bool)
            logits = model.option_logits(trunk, sb, selb, picked)[0]
            probs = torch.softmax(logits, dim=-1)
            actions = [(j,) for j in range(n)]
            pri = [float(probs[j]) for j in range(n)]
            if lo == 0:
                actions.append(())
                pri.append(float(probs[O]))
        else:
            cand = {}
            for _ in range(m_multipick):
                picks, logp, _ = run_pick_loop(model, trunk, sb, selb,
                                               generator=gen)
                key = tuple(picks)
                cand[key] = max(cand.get(key, 0.0),
                                math.exp(min(float(logp), 0.0)))
            g = tuple(_greedy_picks(model, trunk, sb, selb))
            if g not in cand:
                cand[g] = max(cand.values()) if cand else 1.0
            actions = list(cand.keys())
            pri = [cand[a] for a in actions]
    tot = sum(pri) or 1.0
    return actions, [p / tot for p in pri], trunk


def _eval_state(model, obs, seat, deck, belief, tables, gen, m_multipick):
    """Featurize a search obs from its acting seat; return actions, priors,
    and public_value FROM THAT SEAT's perspective."""
    import torch

    from .featurize import encode_select, featurize_state
    ts = featurize_state(obs, seat, deck, belief, tables)
    es = encode_select(obs, ts, tables)
    actions, priors, trunk = propose_actions(model, ts, es, obs["select"],
                                             gen, m_multipick)
    with torch.no_grad():
        v = float(model.public_value(trunk))
    return actions, priors, v


def _select_action(node, c_puct):
    sqrt_total = math.sqrt(sum(node.N) + 1)
    best_i, best_s = 0, float("-inf")
    for i in range(len(node.actions)):
        q = node.W[i] / node.N[i] if node.N[i] else 0.0
        s = q + c_puct * node.P[i] * sqrt_total / (1 + node.N[i])
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def _simulate(root, me, my_deck, opp_decklist, model, tables, session, gen,
              cfg):
    """One PUCT simulation: descend, expand nodes, back up one my-phase value.

    A leaf's public_value is only read at a my-seat node: the value head
    has a turn-phase level shift (turn-start |v|~0.9 vs ~0.6 mid-turn for
    comparable advantage), so comparing it across phases biased the search
    toward passive, turn-ending lines. An opponent node is expanded for its
    actions/priors only (its value is discarded) and the descent continues
    through it by PUCT, so every backed-up value is my-phase or an exact
    terminal/dead-edge. A depth bound backs up a neutral 0.0 instead of
    hanging if that never happens (it shouldn't: engine turns are finite).
    """
    path = []
    node = root
    v_me = 0.0
    for _ in range(64):
        if node.term_v is not None:
            v_me = node.term_v
            break
        if node.actions is None:            # unexpanded: evaluate
            seat = node.seat
            deck = my_deck if seat == me else opp_decklist
            belief = node.trk[seat].snapshot()
            try:
                actions, priors, v = _eval_state(
                    model, node.obs, seat, deck, belief, tables, gen,
                    cfg.m_multipick)
            except Exception:
                actions = []
            if not actions:
                node.term_v = 0.0
                v_me = 0.0
                break
            node.actions = actions
            node.P = priors
            node.N = [0] * len(actions)
            node.W = [0.0] * len(actions)
            if seat == me:                  # my-phase leaf: stop here
                v_me = v
                break
            # opponent node: v is off-phase -- discard it and keep
            # descending (below) via PUCT on the freshly expanded node
        a = _select_action(node, cfg.c_puct)
        path.append((node, a))
        child = node.children.get(a)
        if child is None:
            nxt = session.step(node.sid, list(node.actions[a]))
            if nxt is None:                 # engine refused: neutral dead edge
                child = _Node(-1, None, node.seat, node.trk, term_v=0.0)
            else:
                sid, obs = nxt
                res = obs["current"]["result"]
                if res != -1:
                    tv = 1.0 if res == me else (0.0 if res == 2 else -1.0)
                    child = _Node(sid, obs, obs["current"]["yourIndex"],
                                  node.trk, term_v=tv)
                else:
                    child = _Node(sid, obs, obs["current"]["yourIndex"],
                                  _child_trackers(node.trk, obs))
            node.children[a] = child
        node = child
    else:
        v_me = 0.0                          # depth bound exhausted: neutral
    for n, a in path:
        n.N[a] += 1
        n.W[a] += v_me if n.seat == me else -v_me
    return True


def _vote_counts(roots):
    """Summed visit and value counters per root pick-tuple across trees."""
    votes, wsum = Counter(), Counter()
    for root, _ in roots:
        if not root.actions:
            continue
        for j, a in enumerate(root.actions):
            votes[a] += root.N[j]
            wsum[a] += root.W[j]
    return votes, wsum


def _vote(roots):
    """Root pick across trees: max summed visits, ties by mean value."""
    votes, wsum = _vote_counts(roots)
    if not votes:
        return None
    return max(votes, key=lambda a: (votes[a],
                                     wsum[a] / votes[a] if votes[a] else float("-inf")))


def search_move(obs, me, my_deck, tracker, model, tables, session, cfg, rng,
                gen, tslice):
    """Search-chosen pick list for the agent's current obs, or None.

    None means "no answer" (caller falls back to raw policy). Never raises;
    always search_end()s the arena before returning.
    """
    import torch
    t0 = time.perf_counter()
    stats = MoveStats()
    try:
        from .determinize import (filler_determinization,
                                  sample_determinization)
        from .featurize import encode_select, featurize_state
        select = obs["select"]
        belief = tracker.snapshot()
        ts = featurize_state(obs, me, my_deck, belief, tables)
        es = encode_select(obs, ts, tables)
        actions, priors, trunk = propose_actions(model, ts, es, select, gen,
                                                 cfg.m_multipick)
        if len(actions) <= 1:
            stats.reason = "single-action"
            return (list(actions[0]) if actions else None), stats
        with torch.no_grad():
            dl = model.aux_decklist(trunk)[0].numpy()
            hd = model.aux_hand(trunk)[0].numpy()
        roots = []
        for _ in range(cfg.k_trees):
            det = sample_determinization(obs, me, my_deck, belief, dl, hd,
                                         tables, rng)
            got = session.begin(obs, det)
            if got is None:
                det = filler_determinization(obs, me, my_deck, tables, rng)
                got = session.begin(obs, det)
            if got is None:
                continue
            sid, robs = got
            trk_me, trk_opp = _clone(tracker), BeliefTracker(1 - me)
            pair = (trk_me, trk_opp) if me == 0 else (trk_opp, trk_me)
            root = _Node(sid, robs, me, pair)
            root.actions = list(actions)
            root.P = list(priors)
            root.N = [0] * len(actions)
            root.W = [0.0] * len(actions)
            roots.append((root, det.opp_decklist))
        stats.trees = len(roots)
        if not roots:
            stats.reason = "no-roots"
            return None, stats
        max_sims = cfg.k_trees * cfg.sims_per_tree
        i = 0
        while (time.perf_counter() - t0) < tslice and stats.sims < max_sims:
            root, odl = roots[i % len(roots)]
            _simulate(root, me, my_deck, odl, model, tables, session, gen,
                      cfg)
            stats.sims += 1
            i += 1
        if stats.sims == 0:
            stats.reason = "no-sims"
            return None, stats
        votes, wsum = _vote_counts(roots)
        if not votes:
            stats.reason = "no-vote"
            return None, stats
        stats.root_actions = list(votes.keys())
        stats.root_visits = [int(votes[a]) for a in stats.root_actions]
        best = max(votes, key=lambda a: (votes[a],
                                         wsum[a] / votes[a] if votes[a] else float("-inf")))
        stats.searched = True
        return list(best), stats
    except Exception:
        stats.reason = "error"
        return None, stats
    finally:
        stats.elapsed = time.perf_counter() - t0
        try:
            session.end()
        except Exception:
            pass
