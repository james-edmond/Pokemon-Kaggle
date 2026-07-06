import csv
import os
import random
from functools import lru_cache

from .engine import BattleSession, engine_dir, random_picks


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    if "(" in s:
        s = s[: s.index("(")].strip()
    return " ".join(s.split())


@lru_cache(maxsize=1)
def card_name_index() -> dict:
    path = os.path.join(os.path.dirname(engine_dir()), "..", "EN_Card_Data.csv")
    path = os.path.normpath(path)
    idx = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = _norm(row["Card Name"])
            cid = int(row["Card ID"])
            idx.setdefault(key, cid)  # first ID for a given name (reprints share function)
    return idx


def resolve(name):
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        return int(name)
    return card_name_index().get(_norm(name))


def deck_from_counts(counts) -> list:
    deck = []
    for name, n in counts:
        cid = resolve(name)
        if cid is None:
            raise KeyError(f"unresolved card: {name!r}")
        deck.extend([cid] * n)
    return deck


def is_legal(deck):
    try:
        s = BattleSession(list(deck), list(deck))
    except ValueError as e:
        return False, str(e)
    else:
        s.close()
        return True, ""


def is_playable(deck, n_games: int = 6, seed: int = 0):
    seat_wins = [0, 0]
    rng = random.Random(seed)
    for _ in range(n_games):
        try:
            s = BattleSession(list(deck), list(deck))
        except ValueError as e:
            return False, f"illegal: {e}"
        try:
            n = 0
            while not s.done:
                n += 1
                if n > 5000:
                    return False, "did not terminate"
                s.select(random_picks(s.obs, rng))
            if s.result in (0, 1):
                seat_wins[s.result] += 1
        finally:
            s.close()
    if seat_wins[0] == 0 or seat_wins[1] == 0:
        return False, f"degenerate outcomes {seat_wins}"
    return True, ""


def validate(deck):
    ok, why = is_legal(deck)
    if not ok:
        return False, why
    return is_playable(deck)
