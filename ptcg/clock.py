"""Wall-clock budget management for inference-time search.

Pure logic: no torch, no engine imports. All budgets are seconds of
measured wall time so the same code self-calibrates to any CPU (Kaggle
included). The bank is PROCESS-LIFETIME: spent time only accumulates, so
a process hosting several games can never reset its budget upward
mid-episode.
"""

OPT_ATTACK = 13  # cg OptionType.ATTACK
OPT_SKILL = 15   # cg OptionType.SKILL: "select the order of card skills"

# select.context values where the pick SEQUENCE may itself encode a decision
# (ordering/placement), not just which items are taken -- a take-all cannot
# be collapsed to ascending indices here, so defer to policy/search instead.
CTX_TO_DECK = 9
CTX_TO_DECK_BOTTOM = 10
CTX_SKILL_ORDER = 34
_ORDER_SEMANTIC_CONTEXTS = (CTX_TO_DECK, CTX_TO_DECK_BOTTOM, CTX_SKILL_ORDER)


def forced_picks(select):
    """The single legal pick-list for a trivial select, else None.

    Trivial = exactly one legal pick-list exists:
      one option with at least one pick required        -> [0]
      must take every option (min == max == len(option)) -> [0..n-1]
    nopt==1 with minCount==0 is a real choice ([] vs [0]), not trivial.

    Take-all (min == max == n) is only trivial when pick ORDER cannot
    matter. Some selects are order-semantic -- the context is one of
    _ORDER_SEMANTIC_CONTEXTS, or an option is type SKILL (order of skills
    to resolve) -- and for those, forcing ascending order would silently
    override a real decision the policy/search should make; return None.
    """
    try:
        options = select["option"]
        n = len(options)
        lo, hi = int(select["minCount"]), int(select["maxCount"])
    except Exception:
        return None
    if n == 1 and lo >= 1 and hi >= 1:
        return [0]
    if n > 0 and lo == hi == n:
        try:
            if select.get("context") in _ORDER_SEMANTIC_CONTEXTS:
                return None
            if any(isinstance(o, dict) and o.get("type") == OPT_SKILL
                   for o in options):
                return None
        except Exception:
            return None
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
