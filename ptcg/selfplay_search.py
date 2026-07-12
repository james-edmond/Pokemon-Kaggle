"""Search-vs-search self-play with expert-iteration recording.

Both seats pick moves with the phase-4 search under a SIMS budget
(tslice=inf so k_trees*sims_per_tree binds): data strength is
machine-load-independent. Each recorded move stores the root's candidate
pick-tuples and RAW cross-tree visit counts (the trainer normalizes, so
temperature stays a train-time knob), plus the public/privileged
featurizations in phase-2 tensor conventions. Moves without a usable
distribution (forced picks, single-action shortcuts, search fallbacks)
are recorded value-only (actions=None): every state still trains the
value and aux heads. record=False skips recording entirely (gate games).
"""
import random
from dataclasses import dataclass

import torch

from .action import sample_select
from .clock import forced_picks
from .engine import BattleSession
from .featurize import (FEATURIZER_VERSION, encode_select,
                        featurize_privileged, featurize_state)
from .mcts import search_move
from .simsearch import SearchSession
from .tracker import BeliefTracker


@dataclass
class EIStep:
    player: int
    state: object            # TokenizedState (public, acting seat)
    esel: object             # EncodedSelect
    priv_state: object       # TokenizedState (privileged)
    actions: list = None     # list[tuple[int,...]] root candidates, or None
    visits: list = None      # raw visit counts aligned with actions, or None


@dataclass
class EIGame:
    steps: list
    result: int
    rewards: tuple
    decks: tuple
    featurizer_version: int = FEATURIZER_VERSION


def sample_deck_pair(rng, mirror_frac=0.3):
    """Two portfolio deck names: mirror with prob mirror_frac, else distinct."""
    from .decks import PORTFOLIO
    names = sorted(PORTFOLIO)
    a = rng.choice(names)
    if rng.random() < mirror_frac:
        return (a, a)
    return (a, rng.choice([n for n in names if n != a]))


def play_search_game(net0, net1, deck_names, tables, *, cfg, rng, gen,
                     session=None, record=True, step_cap=5000):
    """One battle where seat 0 plays net0 and seat 1 plays net1, both with
    search. Returns an EIGame (steps empty when record=False)."""
    from .decks import deck as deck_by_name
    decks = (deck_by_name(deck_names[0]), deck_by_name(deck_names[1]))
    nets = (net0, net1)
    session = session or SearchSession()
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    last_obs = [s.obs, s.obs]
    seen = [False, False]
    steps = []
    try:
        while not s.done:
            if len(steps) >= step_cap:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            last_obs[me] = s.obs
            seen[me] = True
            trackers[me].update(s.obs.get("logs", []))
            ts = es = pv = None
            if record:
                ts = featurize_state(s.obs, me, decks[me],
                                     trackers[me].snapshot(), tables)
                es = encode_select(s.obs, ts, tables)
                # a seat that has not yet acted has no obs of its own: its
                # slot in last_obs holds the other seat's obs, where its hand
                # is None. Source that hand from VisualizeData.
                vcur = s.viz_current()
                viz_hands = None
                if not (seen[0] and seen[1]):
                    vp = vcur.get("players") or []
                    if len(vp) == 2:
                        viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
                pv = featurize_privileged(last_obs[0], last_obs[1], decks,
                                          tables, viz=vcur,
                                          viz_hands=viz_hands)
            actions = visits = None
            fp = forced_picks(s.obs["select"])
            if fp is not None:
                picks = fp
            else:
                picks, st = search_move(
                    s.obs, me, decks[me], trackers[me], nets[me], tables,
                    session, cfg, rng, gen, tslice=float("inf"))
                if picks is None:
                    if ts is None:
                        ts = featurize_state(s.obs, me, decks[me],
                                             trackers[me].snapshot(), tables)
                        es = encode_select(s.obs, ts, tables)
                    d = sample_select(nets[me], ts, es, gen)
                    picks = d.picks
                elif (st.searched and st.root_actions is not None
                        and len(st.root_actions) >= 2
                        and sum(st.root_visits) > 0):
                    actions = list(st.root_actions)
                    visits = [int(v) for v in st.root_visits]
            if record:
                steps.append(EIStep(me, ts, es, pv, actions, visits))
            s.select(list(picks))
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    return EIGame(steps, r, rewards, decks=(list(decks[0]), list(decks[1])))
