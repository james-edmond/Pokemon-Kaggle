"""Wall-clock budget management for inference-time search.

Pure logic: no torch, no engine imports. All budgets are seconds of
measured wall time so the same code self-calibrates to any CPU (Kaggle
included). The bank is PROCESS-LIFETIME: spent time only accumulates, so
a process hosting several games can never reset its budget upward
mid-episode.
"""

OPT_ATTACK = 13  # cg OptionType.ATTACK


def forced_picks(select):
    """The single legal pick-list for a trivial select, else None.

    Trivial = exactly one legal pick-list exists:
      one option with at least one pick required        -> [0]
      must take every option (min == max == len(option)) -> [0..n-1]
    nopt==1 with minCount==0 is a real choice ([] vs [0]), not trivial.
    """
    try:
        n = len(select["option"])
        lo, hi = int(select["minCount"]), int(select["maxCount"])
    except Exception:
        return None
    if n == 1 and lo >= 1 and hi >= 1:
        return [0]
    if n > 0 and lo == hi == n:
        return list(range(n))
    return None


class SearchClock:
    def __init__(self, bank_s=480.0, floor_s=60.0, cap_s=20.0,
                 expected_total_moves=80):
        self.bank_s = float(bank_s)
        self.floor_s = float(floor_s)
        self.cap_s = float(cap_s)
        self.expected_total_moves = int(expected_total_moves)
        self.spent = 0.0            # process-lifetime, never resets
        self.moves_this_game = 0

    @property
    def remaining(self):
        return self.bank_s - self.spent

    def new_game(self):
        self.moves_this_game = 0

    def note_move(self):
        self.moves_this_game += 1

    def slice_for(self, select):
        """Seconds this move may spend searching (0.0 = don't search)."""
        if forced_picks(select) is not None:
            return 0.0
        if self.remaining < self.floor_s:
            return 0.0
        exp_rem = max(20, self.expected_total_moves - self.moves_this_game)
        imp = 1.0
        try:
            opts = select["option"]
            if len(opts) >= 6 or any(
                    isinstance(o, dict) and o.get("type") == OPT_ATTACK
                    for o in opts):
                imp = 1.5
        except Exception:
            pass
        return max(0.0, min(self.cap_s, self.remaining / exp_rem * imp))

    def charge(self, seconds):
        self.spent += max(0.0, float(seconds))
