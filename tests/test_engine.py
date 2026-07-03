import pytest
from ptcg.engine import BattleSession, random_picks, load_sample_deck


def test_sample_deck_loads():
    deck = load_sample_deck()
    assert len(deck) == 60
    assert all(isinstance(c, int) and c > 0 for c in deck)


def test_full_random_game(sample_deck, rng):
    s = BattleSession(sample_deck, list(sample_deck))
    try:
        n = 0
        while not s.done:
            assert n < 5000, "selection cap hit"
            s.select(random_picks(s.obs, rng))
            n += 1
        assert s.result in (0, 1, 2)
        assert 5 <= n <= 1000  # benchmark observed 13..219
    finally:
        s.close()


def test_one_session_per_process(sample_deck, rng):
    s = BattleSession(sample_deck, list(sample_deck))
    try:
        with pytest.raises(RuntimeError):
            BattleSession(sample_deck, list(sample_deck))
    finally:
        s.close()
    s2 = BattleSession(sample_deck, list(sample_deck))  # reopen after close works
    s2.close()
