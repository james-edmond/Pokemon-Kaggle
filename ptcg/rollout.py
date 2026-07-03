from dataclasses import dataclass

from .action import SelectDecision, sample_select
from .engine import BattleSession
from .featurize import (EncodedSelect, TokenizedState, encode_select,
                        featurize_privileged, featurize_state)
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


def play_game(model, decks, tables, generator=None, step_cap=5000):
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    last_obs = [s.obs, s.obs]
    steps = []
    try:
        while not s.done:
            if len(steps) >= step_cap:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            last_obs[me] = s.obs
            trackers[me].update(s.obs.get("logs", []))
            ts = featurize_state(s.obs, me, decks[me], trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            pv = featurize_privileged(last_obs[0], last_obs[1], decks, tables)
            d = sample_select(model, ts, es, generator)
            steps.append(Step(me, ts, es, d.picks, d.logprob, pv))
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    return Episode(steps, r, rewards)
