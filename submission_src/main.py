import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
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
_STATE = {"tracker": None, "me": None}


def _ensure_model():
    global _TABLES, _MODEL, _GEN
    if _MODEL is None:
        import torch
        from ptcg.cards import build_tables
        from ptcg.model import PolicyModel, student_config
        _TABLES = build_tables()
        m = PolicyModel(student_config(_TABLES))
        m.load_state_dict(torch.load(os.path.join(_HERE, "policy.pt"),
                                     map_location="cpu"))
        m.eval()
        _MODEL = m
        _GEN = torch.Generator().manual_seed(0)


def _fallback(obs_dict):
    sel = obs_dict["select"]
    return random.sample(range(len(sel["option"])), sel["maxCount"])


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
        import torch
        from ptcg.action import sample_select
        from ptcg.featurize import encode_select, featurize_state
        from ptcg.tracker import BeliefTracker
        _ensure_model()
        me = obs_dict["current"]["yourIndex"]
        if _STATE["tracker"] is None or _STATE["me"] != me:
            _STATE["tracker"] = BeliefTracker(me)
            _STATE["me"] = me
        _STATE["tracker"].update(obs_dict.get("logs", []))
        ts = featurize_state(obs_dict, me, _DECK, _STATE["tracker"].snapshot(), _TABLES)
        es = encode_select(obs_dict, ts, _TABLES)
        with torch.no_grad():
            d = sample_select(_MODEL, ts, es, _GEN)
        picks = list(d.picks)
        return picks if _is_legal(picks, obs_dict["select"]) else _fallback(obs_dict)
    except Exception:
        return _fallback(obs_dict)
