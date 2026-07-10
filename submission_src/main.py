import os
import random
import sys
import time


def _agent_dir():
    # Kaggle exec()s main.py with NO __file__ defined; the agent's files live
    # in its own directory (typically /kaggle_simulations/agent). Resolve
    # robustly.
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
    os.environ.setdefault("PTCG_ENGINE_DIR", _HERE)
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

# --- inference-time search (phase 4) ---
_SEARCH_ENABLED = True
_SESSION = None
_CLOCK = None
_SCFG = None
_RNG = None
_TELEM = {"games": 0, "moves": 0, "searched": 0, "sims": 0,
          "fallbacks": 0, "search_time": 0.0}


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


def _ensure_search():
    global _SESSION, _CLOCK, _SCFG, _RNG
    if _CLOCK is None:
        from ptcg.clock import SearchClock
        from ptcg.mcts import SearchConfig
        from ptcg.simsearch import SearchSession
        _SESSION = SearchSession()
        _CLOCK = SearchClock()
        _SCFG = SearchConfig()
        _RNG = random.Random(0)


def _configure_search(bank_s=None, floor_s=None, cap_s=None, k_trees=None,
                      sims_per_tree=None):
    """Test/eval knob: override clock + search sizes."""
    _ensure_search()
    for k, v in (("bank_s", bank_s), ("floor_s", floor_s), ("cap_s", cap_s)):
        if v is not None:
            setattr(_CLOCK, k, float(v))
    if k_trees is not None:
        _SCFG.k_trees = int(k_trees)
    if sims_per_tree is not None:
        _SCFG.sims_per_tree = int(sims_per_tree)


def _reseed(seed):
    """Test/eval knob: reseed sampling + search RNGs."""
    _ensure_model()
    _ensure_search()
    _GEN.manual_seed(int(seed))
    _RNG.seed(int(seed))


def _flush_telemetry():
    if _TELEM["moves"]:
        print("[agent] games=%d moves=%d searched=%d sims=%d fallbacks=%d "
              "search_time=%.1fs" % (_TELEM["games"], _TELEM["moves"],
                                     _TELEM["searched"], _TELEM["sims"],
                                     _TELEM["fallbacks"],
                                     _TELEM["search_time"]),
              file=sys.stderr, flush=True)


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


def _try_search(obs_dict, me):
    """Search-chosen picks or None. Never raises."""
    try:
        if not _SEARCH_ENABLED:
            return None
        sbi = obs_dict.get("search_begin_input")
        if not isinstance(sbi, str) or not sbi:
            return None
        _ensure_search()
        sel = obs_dict["select"]
        tslice = _CLOCK.slice_for(sel)
        if tslice <= 0.0:
            return None
        from ptcg.mcts import search_move
        picks, stats = search_move(obs_dict, me, _DECK, _STATE["tracker"],
                                   _MODEL, _TABLES, _SESSION, _SCFG, _RNG,
                                   _GEN, tslice)
        _CLOCK.charge(stats.elapsed)
        _TELEM["search_time"] += stats.elapsed
        if stats.searched:
            _TELEM["searched"] += 1
            _TELEM["sims"] += stats.sims
        if picks is not None and _is_legal(picks, sel):
            return picks
        _TELEM["fallbacks"] += 1
        return None
    except Exception:
        _TELEM["fallbacks"] += 1
        return None


def agent(obs_dict):
    if obs_dict.get("select") is None:
        _flush_telemetry()
        _STATE["tracker"] = None
        _STATE["me"] = None
        _TELEM["games"] += 1
        if _CLOCK is not None:
            _CLOCK.new_game()
        return list(_DECK)
    try:
        from ptcg.clock import forced_picks
        from ptcg.tracker import BeliefTracker
        me = obs_dict["current"]["yourIndex"]
        if _STATE["tracker"] is None or _STATE["me"] != me:
            _STATE["tracker"] = BeliefTracker(me)
            _STATE["me"] = me
        _STATE["tracker"].update(obs_dict.get("logs", []))
        _TELEM["moves"] += 1
        if _CLOCK is not None:
            _CLOCK.note_move()
        fp = forced_picks(obs_dict["select"])
        if fp is not None:
            return fp
        import torch
        from ptcg.action import sample_select
        from ptcg.featurize import encode_select, featurize_state
        _ensure_model()
        if _CLOCK is None and _SEARCH_ENABLED:
            _ensure_search()
            _CLOCK.note_move()
        picks = _try_search(obs_dict, me)
        if picks is not None:
            return picks
        ts = featurize_state(obs_dict, me, _DECK, _STATE["tracker"].snapshot(),
                             _TABLES)
        es = encode_select(obs_dict, ts, _TABLES)
        with torch.no_grad():
            d = sample_select(_MODEL, ts, es, _GEN)
        picks = [int(p) for p in d.picks]
        return picks if _is_legal(picks, obs_dict["select"]) else _fallback(obs_dict)
    except Exception:
        return _fallback(obs_dict)
