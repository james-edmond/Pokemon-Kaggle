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


def test_fallback_never_raises_on_malformed_select():
    mod = _load_agent()
    # a non-None select missing "current"/"maxCount" must still yield a list, never raise
    assert isinstance(mod.agent({"select": {"option": [0, 1]}, "logs": []}), list)
    assert mod.agent({"select": {}, "logs": []}) == []
    # maxCount > len(option) is clamped to a legal distinct pick
    picks = mod._fallback({"select": {"option": [0, 1], "minCount": 1, "maxCount": 5}})
    assert isinstance(picks, list) and len(picks) == 2 and len(set(picks)) == 2
    # totally malformed select never raises
    assert mod._fallback({"select": {}}) == []


def test_agent_loads_under_exec_without_dunder_file():
    # Kaggle loads the agent via exec(code, env) with NO __file__ defined.
    import os
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "submission_src", "main.py")
    code = open(src).read()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(src))  # so _agent_dir() finds deck.csv in cwd
    try:
        ns = {}
        exec(compile(code, "main.py", "exec"), ns)  # must NOT raise NameError
        deck = ns["agent"]({"select": None, "current": None, "logs": []})
        assert isinstance(deck, list) and len(deck) == 60
    finally:
        os.chdir(cwd)


def test_search_disabled_flag_and_knobs_exist():
    mod = _load_agent()
    assert mod._SEARCH_ENABLED is True
    mod._configure_search(cap_s=0.5, k_trees=2, sims_per_tree=4, bank_s=100.0)
    assert mod._CLOCK.cap_s == 0.5 and mod._CLOCK.bank_s == 100.0
    assert mod._SCFG.k_trees == 2 and mod._SCFG.sims_per_tree == 4
    mod._reseed(7)                       # must not raise
    assert set(mod._TELEM) >= {"games", "moves", "searched", "sims",
                               "fallbacks", "search_time"}


def test_trivial_select_fast_path_no_model():
    mod = _load_agent()
    # forced single option answered instantly, without loading the model
    obs = {"select": {"option": [{"type": 0}], "minCount": 1, "maxCount": 1},
           "current": {"yourIndex": 0, "players": [{}, {}]}, "logs": []}
    assert mod.agent(obs) == [0]
    assert mod._MODEL is None            # fast path never touched torch
    obs2 = {"select": {"option": [{"type": 0}] * 3, "minCount": 3,
                       "maxCount": 3},
            "current": {"yourIndex": 0, "players": [{}, {}]}, "logs": []}
    assert mod.agent(obs2) == [0, 1, 2]


def test_search_off_agent_still_never_raises():
    mod = _load_agent()
    mod._SEARCH_ENABLED = False
    obs = {"select": {"option": list(range(5)), "minCount": 1, "maxCount": 2},
           "logs": []}
    picks = mod.agent(obs)
    assert isinstance(picks, list) and 1 <= len(picks) <= 2


def test_missing_search_begin_input_skips_search():
    mod = _load_agent()
    # search enabled but no sbi: must fall through to policy/fallback with
    # no exception (this obs also fails featurization -> random legal)
    obs = {"select": {"option": list(range(4)), "minCount": 1, "maxCount": 1},
           "current": {"yourIndex": 0, "players": [{}, {}]}, "logs": []}
    picks = mod.agent(obs)
    assert isinstance(picks, list) and len(picks) == 1
