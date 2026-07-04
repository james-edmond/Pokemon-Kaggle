import json
import os
import random
import sys

_game = None


def engine_dir() -> str:
    p = os.environ.get("PTCG_ENGINE_DIR")
    if p:
        return p
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo, "pokemon-tcg-ai-battle", "sample_submission", "sample_submission"
    )


def _load_game():
    global _game
    if _game is None:
        d = engine_dir()
        if not os.path.isdir(d):
            raise FileNotFoundError(f"engine dir not found: {d} (set PTCG_ENGINE_DIR)")
        if d not in sys.path:
            sys.path.insert(0, d)
        from cg import game  # loads the native library on import
        _game = game
    return _game


def load_sample_deck() -> list[int]:
    path = os.path.join(engine_dir(), "deck.csv")
    with open(path) as f:
        return [int(line) for line in f.read().split("\n")[:60]]


def random_picks(obs: dict, rng: random.Random) -> list[int]:
    sel = obs["select"]
    return rng.sample(range(len(sel["option"])), sel["maxCount"])


class BattleSession:
    """One battle per process: the engine keeps battle_ptr as global state."""

    _open = False

    def __init__(self, deck0: list[int], deck1: list[int]):
        if BattleSession._open:
            raise RuntimeError("a BattleSession is already open in this process")
        g = _load_game()
        obs, start = g.battle_start(list(deck0), list(deck1))
        if obs is None:
            raise ValueError(
                f"deck rejected: player={start.errorPlayer} type={start.errorType}"
            )
        BattleSession._open = True
        self._g = g
        self.obs = obs

    @property
    def select_player(self) -> int:
        return self.obs["current"]["yourIndex"]

    @property
    def result(self) -> int:
        return self.obs["current"]["result"]

    @property
    def done(self) -> bool:
        return self.result != -1

    def select(self, picks: list[int]) -> dict:
        self.obs = self._g.battle_select(list(picks))
        return self.obs

    def viz_current(self) -> dict:
        """Latest VisualizeData snapshot's full-information `current` dict."""
        return json.loads(self._g.visualize_data())[-1]["current"]

    def close(self) -> None:
        if BattleSession._open:
            self._g.battle_finish()
            BattleSession._open = False
