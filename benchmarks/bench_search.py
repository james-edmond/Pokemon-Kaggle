"""Measurement 5 (stretch): search API throughput (cg.api search_begin/search_step).

Takes a mid-game snapshot (30 random selections in), builds count-correct
hidden-card predictions from the sample deck (basics first, so the opponent
deck prediction always contains a Basic Pokemon), then runs random rollouts:
search_begin -> search_step... until terminal -> search_end -> repeat, for
WINDOW_S seconds. search_step cost inherently includes the json->dataclass
conversion of each returned SearchState.
"""
import random
import time

import common

common.add_engine_to_path()
from cg import game  # noqa: E402
from cg.api import (all_card_data, search_begin, search_end, search_step,  # noqa: E402
                    to_observation_class, CardType)

MIDGAME_SELECTIONS = 30
SEED = 7000
WINDOW_S = 20.0
ROLLOUT_CAP = 5000


def main():
    deck = common.load_deck()
    rng = random.Random(SEED)
    obs, start = game.battle_start(deck, deck)
    for _ in range(MIDGAME_SELECTIONS):
        if obs["current"]["result"] != -1:
            raise RuntimeError("game ended before mid-game snapshot; pick another seed")
        obs = game.battle_select(common.random_action(obs["select"], rng))
    agent_obs = to_observation_class(obs)
    st = agent_obs.current
    yi = st.yourIndex
    me, opp = st.players[yi], st.players[1 - yi]
    print(f"mid-game snapshot after {MIDGAME_SELECTIONS} selections: turn={st.turn} "
          f"yourIndex={yi} myDeck={me.deckCount} oppDeck={opp.deckCount} "
          f"oppHand={opp.handCount} myPrize={len(me.prize)} oppPrize={len(opp.prize)}")

    basics = {c.cardId for c in all_card_data() if c.cardType == CardType.POKEMON and c.basic}
    deck_basics_first = sorted(deck, key=lambda cid: cid not in basics)

    def pred(n):
        reps = (n + len(deck) - 1) // len(deck) if n else 1
        return (deck_basics_first * reps)[:n]

    opp_active = []
    if len(opp.active) > 0 and opp.active[0] is None:
        opp_active = [deck_basics_first[0]]

    def begin():
        return search_begin(agent_obs, pred(me.deckCount), pred(len(me.prize)),
                            pred(opp.deckCount), pred(len(opp.prize)),
                            pred(opp.handCount), opp_active)

    # warmup: one short rollout (also pays AgentStart allocation)
    state = begin()
    for _ in range(5):
        o = state.observation
        if o.current.result != -1:
            break
        state = search_step(state.searchId, rng.sample(range(len(o.select.option)), o.select.maxCount))
    search_end()

    begins = steps = terminals = caps = 0
    t_begin = t_step = 0.0
    state = None
    rollout_steps = 0
    t_all = time.perf_counter()
    while True:
        if state is None:
            t0 = time.perf_counter()
            state = begin()
            t_begin += time.perf_counter() - t0
            begins += 1
            rollout_steps = 0
        o = state.observation
        if o.current.result != -1 or rollout_steps >= ROLLOUT_CAP:
            terminals += o.current.result != -1
            caps += rollout_steps >= ROLLOUT_CAP and o.current.result == -1
            search_end()
            state = None
        else:
            action = rng.sample(range(len(o.select.option)), o.select.maxCount)
            t0 = time.perf_counter()
            state = search_step(state.searchId, action)
            t_step += time.perf_counter() - t0
            steps += 1
            rollout_steps += 1
        elapsed = time.perf_counter() - t_all
        if elapsed >= WINDOW_S and steps > 0:
            break
    game.battle_finish()

    print(f"{steps} search_steps, {begins} search_begins, {terminals} terminals, "
          f"{caps} rollout-cap hits in {elapsed:.2f}s")
    print(f"  search_step:  {steps / t_step:.0f}/s inside call time "
          f"({1e6 * t_step / steps:.1f} us/step incl. json->dataclass decode)")
    print(f"  search_begin: {1e3 * t_begin / begins:.2f} ms/begin (n={begins})")
    print(f"  end-to-end rollout throughput: {steps / elapsed:.0f} steps/s, "
          f"{steps / begins:.1f} steps/rollout")


if __name__ == "__main__":
    main()
