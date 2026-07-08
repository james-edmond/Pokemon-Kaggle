import os
import random
import sys

def _agent_dir():
    # Kaggle exec()s main.py with NO __file__ defined; the agent's files live in
    # its own directory (typically /kaggle_simulations/agent). Resolve robustly.
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(d, "deck.csv")):
            return d
    except NameError:
        pass
    for cand in ("/kaggle_simulations/agent", os.getcwd()):
        if os.path.exists(os.path.join(cand, "deck.csv")):
            return cand
    return os.getcwd()


_HERE = _agent_dir()
if os.path.isdir(os.path.join(_HERE, "cg")):
    os.environ.setdefault("PTCG_ENGINE_DIR", _HERE)  # bundled cg/ lives beside this file
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _read_deck():
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        rows = [r for r in f.read().split("\n") if r.strip()]
    return [int(r) for r in rows[:60]]


_DECK = _read_deck()
_TABLES = None
_MODEL = None
_GEN = None
_GAVE_UP = False
_STATE = {"tracker": None, "me": None}


def _ensure_model():
    global _TABLES, _MODEL, _GEN, _GAVE_UP
    if _MODEL is not None or _GAVE_UP:
        return
    try:
        import pickle
        import torch
        import ptcg.cards  # noqa: F401 -- needed so pickle can reconstruct CardTables
        from ptcg.model import PolicyModel, student_config
        with open(os.path.join(_HERE, "tables.pkl"), "rb") as f:
            _TABLES = pickle.load(f)
        m = PolicyModel(student_config(_TABLES))
        m.load_state_dict(torch.load(os.path.join(_HERE, "policy.pt"),
                                     map_location="cpu"))
        m.eval()
        _MODEL = m
        _GEN = torch.Generator().manual_seed(0)
    except Exception:
        _GAVE_UP = True


def _fallback(obs_dict):
    try:
        sel = obs_dict["select"]
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n))
    except Exception:
        return []


def _is_legal(picks, sel):
    n = len(sel["option"])
    return (isinstance(picks, list)
            and len(set(picks)) == len(picks)
            and all(isinstance(p, int) and 0 <= p < n for p in picks)
            and sel["minCount"] <= len(picks) <= sel["maxCount"])


def agent(obs_dict):
    if obs_dict.get("select") is None:
        _STATE["tracker"] = None
        _STATE["me"] = None
        return list(_DECK)
    try:
        _ensure_model()
        if _MODEL is None:
            return _fallback(obs_dict)
        import torch
        from ptcg.action import sample_select
        from ptcg.featurize import encode_select, featurize_state
        from ptcg.tracker import BeliefTracker
        me = obs_dict["current"]["yourIndex"]
        if _STATE["tracker"] is None or _STATE["me"] != me:
            _STATE["tracker"] = BeliefTracker(me)
            _STATE["me"] = me
        _STATE["tracker"].update(obs_dict.get("logs", []))
        ts = featurize_state(obs_dict, me, _DECK, _STATE["tracker"].snapshot(), _TABLES)
        es = encode_select(obs_dict, ts, _TABLES)
        with torch.no_grad():
            d = sample_select(_MODEL, ts, es, _GEN)
        picks = [int(p) for p in d.picks]
        return picks if _is_legal(picks, obs_dict["select"]) else _fallback(obs_dict)
    except Exception:
        return _fallback(obs_dict)


# Load the model at import (the untimed agent-setup phase on Kaggle) so no timed
# game move pays the one-time init cost. Never breaks the import: on failure the
# agent degrades to legal-random play.
_ensure_model()
