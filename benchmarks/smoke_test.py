"""Smoke test: one full random-vs-random game through the documented cg.game API.

Prints selection count, turn count, result. Caps at SELECTION_CAP selections
and reports loudly if the cap is hit.
"""
import platform
import random
import sys
import time

import common

common.add_engine_to_path()
from cg import game  # noqa: E402  (loads the native library, runs GameInitialize)


def main():
    print(f"python {sys.version}")
    print(f"platform {platform.platform()} machine={platform.machine()}")
    deck = common.load_deck()
    print(f"deck: {len(deck)} cards, first 5: {deck[:5]}")

    rng = random.Random(42)
    t0 = time.perf_counter()
    obs, start = game.battle_start(deck, deck)
    print(f"battle_start: errorPlayer={start.errorPlayer} errorType={start.errorType} "
          f"battlePtr={'ok' if start.battlePtr else 'NULL'}")
    if obs is None:
        raise SystemExit("FAIL: deck validation failed")
    sel = obs["select"]
    print(f"first obs: select type={sel['type']} context={sel['context']} "
          f"minCount={sel['minCount']} maxCount={sel['maxCount']} n_options={len(sel['option'])}")

    selections = 0
    capped = False
    while obs["current"]["result"] == -1:
        if selections >= common.SELECTION_CAP:
            capped = True
            break
        obs = game.battle_select(common.random_action(obs["select"], rng))
        selections += 1
    elapsed = time.perf_counter() - t0
    game.battle_finish()

    if capped:
        print(f"*** SELECTION CAP HIT ({common.SELECTION_CAP}) — game did not terminate ***")
    print(f"selections: {selections}")
    print(f"final turn: {obs['current']['turn']}")
    print(f"result: {obs['current']['result']}  (0/1 = winner index, 2 = draw)")
    print(f"wall: {elapsed:.3f}s  ({1e6 * elapsed / max(selections, 1):.1f} us/selection, full game.py API)")


if __name__ == "__main__":
    main()
