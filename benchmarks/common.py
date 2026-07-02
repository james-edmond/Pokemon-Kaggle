"""Shared helpers for PTCG engine benchmarks.

Stdlib only. Portable across Windows/Linux/macOS: the engine package is
resolved relative to the repo root (parent of this file's directory), and
cg/sim.py picks the right native library per OS at import time.
"""
import ctypes
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
DECK_CSV = ENGINE_DIR / "deck.csv"
SELECTION_CAP = 5000


def add_engine_to_path():
    p = str(ENGINE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_deck() -> list:
    lines = DECK_CSV.read_text().splitlines()
    deck = [int(x) for x in lines[:60]]
    if len(deck) != 60:
        raise ValueError(f"expected 60 cards in {DECK_CSV}, got {len(deck)}")
    return deck


def random_action(select: dict, rng) -> list:
    # Same policy as the sample agent (main.py): always pick exactly maxCount.
    return rng.sample(range(len(select["option"])), select["maxCount"])


def play_game_tier_b(lib, deck: list, rng, cap: int = SELECTION_CAP):
    """One random-vs-random game via raw ctypes + json.loads only (tier b).

    Returns (selections, result, capped). result is 0/1 winner index, 2 draw,
    -999 if the selection cap was hit. Engine Select errors raise, with context.
    """
    cards = deck + deck
    arr = (ctypes.c_int * len(cards))(*cards)
    start = lib.BattleStart(arr)
    ptr = start.battlePtr
    if not ptr:
        raise RuntimeError(
            f"BattleStart failed: errorPlayer={start.errorPlayer} errorType={start.errorType}")
    selections = 0
    try:
        obs = json.loads(lib.GetBattleData(ptr).json)
        while obs["current"]["result"] == -1:
            if selections >= cap:
                return selections, -999, True
            action = random_action(obs["select"], rng)
            sel_arr = (ctypes.c_int * len(action))(*action)
            err = lib.Select(ptr, sel_arr, len(action))
            if err != 0:
                raise RuntimeError(
                    f"lib.Select error {err} at selection {selections} "
                    f"(action={action}, minCount={obs['select']['minCount']}, "
                    f"maxCount={obs['select']['maxCount']}, "
                    f"n_options={len(obs['select']['option'])})")
            selections += 1
            obs = json.loads(lib.GetBattleData(ptr).json)
        return selections, obs["current"]["result"], False
    finally:
        lib.BattleFinish(ptr)


def rss_bytes() -> int:
    """Current resident set size of this process, stdlib only."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        # explicit prototypes required: the -1 pseudo-handle truncates without them
        gpmi = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.WinDLL("psapi").GetProcessMemoryInfo
        gpmi.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        gpmi.restype = ctypes.c_int
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not gpmi(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(f"GetProcessMemoryInfo failed (winerror {ctypes.get_last_error()})")
        return pmc.WorkingSetSize
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        raise OSError("VmRSS not found in /proc/self/status")
    import resource  # macOS fallback: ru_maxrss is PEAK rss in bytes, not current
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def pctl(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(vals) -> dict:
    s = sorted(vals)
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p95": pctl(s, 95), "min": s[0], "max": s[-1]}


def fmt_summary(label, vals, unit=""):
    s = summarize(vals)
    return (f"{label}: n={s['n']} mean={s['mean']:.2f} median={s['median']:.2f} "
            f"p95={s['p95']:.2f} min={s['min']:.2f} max={s['max']:.2f} {unit}")
