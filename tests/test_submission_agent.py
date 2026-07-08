import importlib.util
import os

import pytest

_MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "submission_src", "main.py")


def _load_agent():
    spec = importlib.util.spec_from_file_location("submission_main", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deck_selection_returns_60_ids():
    mod = _load_agent()
    deck = mod.agent({"select": None, "current": None, "logs": []})
    assert isinstance(deck, list) and len(deck) == 60
    assert all(isinstance(c, int) for c in deck)


def test_illegal_or_broken_obs_falls_back_to_legal_pick():
    mod = _load_agent()
    # An obs whose featurization will fail (missing 'current') must still yield a
    # legal selection from the option list, never an exception.
    obs = {"select": {"option": list(range(5)), "minCount": 1, "maxCount": 2},
           "logs": []}
    picks = mod.agent(obs)
    assert isinstance(picks, list)
    assert 1 <= len(picks) <= 2
    assert len(set(picks)) == len(picks)
    assert all(0 <= p < 5 for p in picks)


def test_is_legal_empty_decline_and_bounds():
    mod = _load_agent()
    sel0 = {"option": [0, 1, 2], "minCount": 0, "maxCount": 1}
    assert mod._is_legal([], sel0) is True          # valid decline when minCount==0
    assert mod._is_legal([1], sel0) is True
    sel1 = {"option": [0, 1, 2], "minCount": 1, "maxCount": 2}
    assert mod._is_legal([], sel1) is False         # must pick >=1
    assert mod._is_legal([0, 0], sel1) is False     # duplicate
    assert mod._is_legal([3], sel1) is False        # out of range
    assert mod._is_legal([0, 1, 2], sel1) is False  # exceeds maxCount
