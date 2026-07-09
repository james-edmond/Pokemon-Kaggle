"""Dict-level wrapper over the engine's Search API.

Owns its own agent arena (lib.AgentStart) and calls SearchBegin/SearchStep/
SearchEnd directly, returning observation DICTS (json.loads of the raw
engine JSON) — the same schema battle obs use, so the featurizer consumes
them unchanged. Replicates cg.api.search_begin's pre-call length checks:
the C side reads exactly the required counts from each array, so a short
array would read out of bounds. begin/step return None on any engine
error or invalid input — they never raise; callers fall back.
"""
import ctypes
import json

from .engine import _load_game


class SearchSession:
    def __init__(self):
        self._lib = None
        self._ptr = None

    def ensure_ptr(self):
        """Load the native lib + create the agent arena once. False on failure."""
        if self._ptr:
            return True
        try:
            _load_game()                 # loads native cg, puts it on sys.path
            from cg.sim import lib
            self._lib = lib
            self._ptr = lib.AgentStart()
        except Exception:
            self._ptr = None
        return bool(self._ptr)

    @staticmethod
    def _arr(xs):
        return (ctypes.c_int * len(xs))(*[int(x) for x in xs])

    def begin(self, obs, det, manual_coin=False):
        """Begin a search from the agent's live obs + determinization.

        Returns (search_id, root_obs_dict) or None.
        """
        if not self.ensure_ptr():
            return None
        try:
            sbi = obs.get("search_begin_input")
            if not isinstance(sbi, str) or not sbi:
                return None
            cur = obs["current"]
            me = cur["yourIndex"]
            you, opp = cur["players"][me], cur["players"][1 - me]
            your_deck = [int(x) for x in det.your_deck]
            if (obs.get("select") or {}).get("deck") is not None:
                your_deck = []           # engine already knows our deck here
            elif len(your_deck) < you["deckCount"]:
                return None
            if (len(det.your_prize) < len(you["prize"] or [])
                    or len(det.opp_deck) < opp["deckCount"]
                    or len(det.opp_prize) < len(opp["prize"] or [])
                    or len(det.opp_hand) < opp["handCount"]):
                return None
            active = opp.get("active") or []
            opp_active = [int(x) for x in det.opp_active]
            if active and active[0] is None:
                if not opp_active:
                    return None
            else:
                opp_active = []
            raw = self._lib.SearchBegin(
                self._ptr, sbi.encode("ascii"), len(sbi),
                self._arr(your_deck), self._arr(det.your_prize),
                self._arr(det.opp_deck), self._arr(det.opp_prize),
                self._arr(det.opp_hand), self._arr(opp_active),
                int(manual_coin))
            return self._parse(raw)
        except Exception:
            return None

    def step(self, search_id, picks):
        """Apply picks to a search state. Returns (child_id, obs_dict) or None."""
        if not self._ptr:
            return None
        try:
            raw = self._lib.SearchStep(self._ptr, int(search_id),
                                       self._arr(picks), len(picks))
            return self._parse(raw)
        except Exception:
            return None

    @staticmethod
    def _parse(raw):
        out = json.loads(raw.decode())
        if out.get("error", 1) != 0 or not out.get("state"):
            return None
        st = out["state"]
        return int(st["searchId"]), st["observation"]

    def end(self):
        """Free every state in the arena (memory reused by the next search)."""
        if self._ptr:
            try:
                self._lib.SearchEnd(self._ptr)
            except Exception:
                pass
