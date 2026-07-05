from dataclasses import dataclass

from .action import SelectDecision, sample_select
from .engine import BattleSession
from .featurize import (FEATURIZER_VERSION, EncodedSelect, TokenizedState,
                        encode_select, featurize_privileged, featurize_state)
from .tracker import BeliefTracker


@dataclass
class Step:
    player: int
    state: TokenizedState
    esel: EncodedSelect
    picks: list
    logprob: float
    priv_state: TokenizedState


@dataclass
class Episode:
    steps: list
    result: int
    rewards: tuple
    featurizer_version: int = FEATURIZER_VERSION


def play_game(model, decks, tables, generator=None, step_cap=5000,
              priv_viz=False, obs_log=None):
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
            if obs_log is not None:
                obs_log.append(s.obs)
            last_obs[me] = s.obs
            seen[me] = True
            trackers[me].update(s.obs.get("logs", []))
            ts = featurize_state(s.obs, me, decks[me], trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            # a seat that has not yet acted has no obs of its own: its slot in
            # last_obs holds the other seat's obs, where its hand is None.
            # Source that hand from the engine's full-information VisualizeData.
            vcur = s.viz_current() if (priv_viz or not (seen[0] and seen[1])) else None
            viz_hands = None
            if not (seen[0] and seen[1]) and vcur is not None:
                vp = vcur.get("players") or []
                if len(vp) == 2:
                    viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
            pv = featurize_privileged(last_obs[0], last_obs[1], decks, tables,
                                      viz=vcur if priv_viz else None,
                                      viz_hands=viz_hands)
            d = sample_select(model, ts, es, generator)
            steps.append(Step(me, ts, es, d.picks, d.logprob, pv))
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    return Episode(steps, r, rewards)
