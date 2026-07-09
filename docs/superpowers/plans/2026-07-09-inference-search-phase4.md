# Inference-Time Search (phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add K-tree PUCT inference-time search (engine search API + aux-head determinization + wall-clock budgeting) to the Kaggle submission agent, per the approved spec `docs/superpowers/specs/2026-07-09-inference-search-phase4-design.md`.

**Architecture:** Four new pure-inference modules — `ptcg/clock.py` (time-bank logic), `ptcg/simsearch.py` (dict-level ctypes wrapper over `lib.SearchBegin/SearchStep/SearchEnd`), `ptcg/determinize.py` (belief+aux-head hidden-info sampling), `ptcg/mcts.py` (K determinized PUCT trees, policy priors, `public_value` leaves, negamax backup, root visit vote) — integrated into `submission_src/main.py` behind the existing never-crash fallback ladder.

**Tech Stack:** Python 3.11, torch CPU, numpy, the competition's native `cg` engine (ctypes), pytest.

## Global Constraints

- NEVER modify anything under `pokemon-tcg-ai-battle/` (the `cg` library is imported/called only).
- Tests run on **base python** from the repo root: `python -m pytest tests/<file> -v`. Run **one pytest invocation at a time** (engine battles are per-process global state; sequential battles in one process are fine, concurrent are not). Always `close()` a `BattleSession` in `finally`.
- The Kaggle agent must never raise: search → raw policy → `_fallback` legal-random. Any new code in `main.py` must preserve that ladder and the existing behavior when search is disabled.
- Do not change `ptcg/featurize.py`, `ptcg/tracker.py`, `ptcg/action.py`, `ptcg/model.py`, `ptcg/cards.py`, `ptcg/engine.py` — new modules consume them as-is.
- Commits on `main`, short lowercase messages, **no co-author lines** (project convention overrides any default).
- `submission_src/policy.pt`, `dist/`, `champ/` are gitignored — never `git add` them.
- Budgets are measured **wall time**, never eval counts (Kaggle CPU speed unknown). Kaggle: `actTimeout=0`, 600 s overage bank/agent, ~2000 s episode.
- Windows console: set `PYTHONIOENCODING=utf-8` when running engine scripts (card names break cp1252).
- Card-table facts you may rely on (from `ptcg/cards.py`): `card_row(card_id, n_rows) == card_id + 2` (so row−2 = card id), `PAD_ROW=0, UNK_ROW=1, N_RESERVED=2`; a row is a real card iff `tables.attr[row].any()`; attr flags `A_BASIC=2, A_ACESPEC=8, A_IS_POKEMON=9, A_IS_BASIC_ENERGY=14`. Basic-energy card ids equal their EnergyType value: Grass 1 … Metal 8.
- Probe-verified engine-search facts (do not re-derive): `search_begin` ≈1 ms, `search_step` ≈0.3 ms and returns a **fresh id per call**; the search obs perspective **flips to the acting player** (their hand visible, per-viewer log deltas); supplied deck **order is ignored** (engine shuffles; only the multiset split matters); terminals set `current.result`; multiple roots coexist; `search_end()` frees all; the C side reads exactly the required counts from each array — **length-validate before calling** or you corrupt memory.

---

### Task 1: `ptcg/clock.py` — trivial-select detection + time bank

**Files:**
- Create: `ptcg/clock.py`
- Test: `tests/test_clock.py`

**Interfaces:**
- Consumes: nothing (pure logic; select dicts have keys `option: list`, `minCount: int`, `maxCount: int`, options are dicts with `type`).
- Produces: `forced_picks(select: dict) -> list[int] | None`; `class SearchClock(bank_s=480.0, floor_s=60.0, cap_s=20.0, expected_total_moves=80)` with `.new_game()`, `.note_move()`, `.slice_for(select: dict) -> float`, `.charge(seconds: float)`, `.remaining -> float`, attributes `bank_s/floor_s/cap_s/expected_total_moves/spent/moves_this_game` (all later tasks use these names).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clock.py`:

```python
from ptcg.clock import SearchClock, forced_picks


def _sel(n, lo, hi, types=None):
    opts = [{"type": t} for t in (types or [0] * n)]
    return {"option": opts, "minCount": lo, "maxCount": hi}


def test_forced_picks_single_required_option():
    assert forced_picks(_sel(1, 1, 1)) == [0]


def test_forced_picks_take_all():
    assert forced_picks(_sel(3, 3, 3)) == [0, 1, 2]


def test_forced_picks_real_choices_return_none():
    assert forced_picks(_sel(1, 0, 1)) is None      # [] vs [0] is a real choice
    assert forced_picks(_sel(3, 1, 1)) is None
    assert forced_picks(_sel(3, 2, 2)) is None      # choose WHICH 2 of 3
    assert forced_picks({"option": []}) is None     # malformed: no crash


def test_slice_basics_and_importance():
    c = SearchClock(bank_s=480.0, floor_s=60.0, cap_s=20.0, expected_total_moves=80)
    base = c.slice_for(_sel(3, 1, 1))
    assert abs(base - 6.0) < 1e-9                   # 480/80
    atk = c.slice_for(_sel(3, 1, 1, types=[0, 13, 0]))   # ATTACK option present
    assert abs(atk - 9.0) < 1e-9                    # 1.5x importance
    many = c.slice_for(_sel(6, 1, 1))
    assert abs(many - 9.0) < 1e-9                   # >=6 options -> 1.5x
    assert c.slice_for(_sel(1, 1, 1)) == 0.0        # trivial -> no search


def test_slice_floor_and_cap():
    c = SearchClock(bank_s=480.0, floor_s=60.0, cap_s=20.0, expected_total_moves=80)
    c.charge(425.0)
    assert c.remaining == 55.0
    assert c.slice_for(_sel(3, 1, 1)) == 0.0        # below floor -> search off
    c2 = SearchClock(bank_s=480.0, cap_s=5.0)
    assert c2.slice_for(_sel(3, 1, 1)) == 5.0       # cap binds (480/80=6 > 5)


def test_bank_is_process_lifetime_and_moves_reset():
    c = SearchClock()
    c.charge(100.0)
    for _ in range(30):
        c.note_move()
    assert c.moves_this_game == 30
    c.new_game()
    assert c.moves_this_game == 0
    assert c.spent == 100.0                          # never resets


def test_expected_remaining_moves_floor():
    c = SearchClock(bank_s=480.0, expected_total_moves=80)
    for _ in range(75):
        c.note_move()
    # exp_rem = max(20, 80-75) = 20 -> slice = 480/20 = 24 -> capped at 20
    assert c.slice_for(_sel(3, 1, 1)) == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clock.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'ptcg.clock'`

- [ ] **Step 3: Write the implementation**

Create `ptcg/clock.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clock.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ptcg/clock.py tests/test_clock.py
git commit -m "add search clock: trivial-select fast path and process-lifetime time bank"
```

---

### Task 2: `ptcg/simsearch.py` — dict-level Search API wrapper

**Files:**
- Create: `ptcg/simsearch.py`
- Test: `tests/test_simsearch.py`

**Interfaces:**
- Consumes: `ptcg.engine._load_game()` (loads native cg, puts it on sys.path); `cg.sim.lib` ctypes symbols (`AgentStart`, `SearchBegin`, `SearchStep`, `SearchEnd` — note `SearchStep`'s id argtype is c_int64); duck-typed determinization objects exposing `.your_deck/.your_prize/.opp_deck/.opp_prize/.opp_hand/.opp_active` (lists of int).
- Produces: `class SearchSession` with `.ensure_ptr() -> bool`, `.begin(obs: dict, det, manual_coin=False) -> tuple[int, dict] | None`, `.step(search_id: int, picks: list[int]) -> tuple[int, dict] | None`, `.end() -> None`. Obs dicts have the live-battle schema (`select/logs/current`). **Never raises** from begin/step/end.

- [ ] **Step 1: Write the failing test**

Create `tests/test_simsearch.py`:

```python
import random
from types import SimpleNamespace

from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.simsearch import SearchSession


def _truth_det(sess, obs):
    """Determinization from visualize_data ground truth (test-only)."""
    viz = sess.viz_current()
    me = obs["current"]["yourIndex"]
    vme, vopp = viz["players"][me], viz["players"][1 - me]
    ids = lambda z: [c["id"] for c in z]
    opp_active = []
    oa = obs["current"]["players"][1 - me].get("active") or []
    if oa and oa[0] is None:
        opp_active = [ids(vopp["active"])[0]]
    return SimpleNamespace(
        your_deck=ids(vme["deck"]), your_prize=ids(vme["prize"]),
        opp_deck=ids(vopp["deck"]), opp_prize=ids(vopp["prize"]),
        opp_hand=ids(vopp["hand"]), opp_active=opp_active)


def test_search_session_round_trip_and_errors():
    deck = load_sample_deck()
    sess = BattleSession(deck, deck)
    rng = random.Random(7)
    try:
        for _ in range(30):
            if sess.done:
                break
            sess.select(random_picks(sess.obs, rng))
        assert not sess.done
        obs = sess.obs
        det = _truth_det(sess, obs)
        ss = SearchSession()
        assert ss.ensure_ptr() is True

        # begin: root select mirrors the live select
        got = ss.begin(obs, det)
        assert got is not None
        sid, robs = got
        assert len(robs["select"]["option"]) == len(obs["select"]["option"])
        assert robs["current"]["yourIndex"] == obs["current"]["yourIndex"]

        # two roots coexist; stepping the first still works
        got2 = ss.begin(obs, det)
        assert got2 is not None and got2[0] != sid
        sel = robs["select"]
        child = ss.step(sid, [0] if sel["minCount"] >= 1 else [])
        assert child is not None and child[0] not in (sid, got2[0])

        # illegal picks -> None, not an exception
        assert ss.step(sid, [len(sel["option"]) + 5]) is None

        # walk to terminal; stepping past it -> None
        node_id, nobs = child
        r = random.Random(3)
        for _ in range(400):
            if nobs["current"]["result"] != -1:
                break
            s = nobs["select"]
            k = r.randint(s["minCount"], s["maxCount"])
            nxt = ss.step(node_id, r.sample(range(len(s["option"])), k))
            assert nxt is not None
            node_id, nobs = nxt
        assert nobs["current"]["result"] != -1
        assert ss.step(node_id, [0]) is None

        # too-short arrays are rejected BEFORE the C call
        bad = SimpleNamespace(your_deck=[], your_prize=det.your_prize,
                              opp_deck=det.opp_deck, opp_prize=det.opp_prize,
                              opp_hand=det.opp_hand, opp_active=det.opp_active)
        assert ss.begin(obs, bad) is None

        # obs without search_begin_input -> None
        stripped = dict(obs)
        stripped.pop("search_begin_input", None)
        assert ss.begin(stripped, det) is None

        ss.end()
        assert ss.step(sid, [0]) is None   # released arena -> None, no raise
    finally:
        sess.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simsearch.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'ptcg.simsearch'`

- [ ] **Step 3: Write the implementation**

Create `ptcg/simsearch.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simsearch.py -v`
Expected: 1 passed (takes a few seconds — real engine)

- [ ] **Step 5: Commit**

```bash
git add ptcg/simsearch.py tests/test_simsearch.py
git commit -m "add simsearch: dict-level wrapper over the engine search api"
```

---

### Task 3: `ptcg/determinize.py` — hidden-info sampling

**Files:**
- Create: `ptcg/determinize.py`
- Test: `tests/test_determinize.py`

**Interfaces:**
- Consumes: `ptcg.cards` constants (`N_RESERVED, card_row, A_BASIC, A_ACESPEC, A_IS_POKEMON, A_IS_BASIC_ENERGY`), `tables.attr` / `tables.n_rows`, `BeliefSnapshot` duck-typed (`.opp_hand/.opp_deck/.opp_hidden_pool` dicts), numpy arrays `dl_lam`/`hd_lam` of shape `(tables.n_rows,)` or `None`.
- Produces: `@dataclass(frozen=True) Determinization(your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active, opp_decklist)` (all `list[int]`); `sample_determinization(obs, me, my_deck, belief, dl_lam, hd_lam, tables, rng) -> Determinization`; `filler_determinization(obs, me, my_deck, tables, rng) -> Determinization`. **Never raises**; exact zone counts whenever the obs is well-formed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_determinize.py`:

```python
import random
from collections import Counter

import numpy as np

from ptcg.cards import build_tables
from ptcg.determinize import (Determinization, filler_determinization,
                              sample_determinization)
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.simsearch import SearchSession
from ptcg.tracker import BeliefTracker


def _advance(sess, tracker, me, rng, n):
    for _ in range(n):
        if sess.done:
            break
        if sess.obs["current"]["yourIndex"] == me:
            tracker.update(sess.obs.get("logs") or [])
        sess.select(random_picks(sess.obs, rng))


def _counts_ok(det, obs, me):
    cur = obs["current"]
    you, opp = cur["players"][me], cur["players"][1 - me]
    assert len(det.your_deck) == you["deckCount"]
    assert len(det.your_prize) == len(you["prize"] or [])
    assert len(det.opp_deck) == opp["deckCount"]
    assert len(det.opp_prize) == len(opp["prize"] or [])
    assert len(det.opp_hand) == opp["handCount"]
    assert len(det.opp_decklist) == 60
    for c in det.your_deck + det.your_prize + det.opp_deck + det.opp_prize + det.opp_hand:
        assert isinstance(c, int) and c >= 1


def test_determinizations_consistent_and_engine_accepted():
    tables = build_tables()
    deck = load_sample_deck()
    sess = BattleSession(deck, deck)
    rng = random.Random(11)
    try:
        me = sess.obs["current"]["yourIndex"]
        tracker = BeliefTracker(me)
        _advance(sess, tracker, me, rng, 25)
        assert not sess.done
        # park on one of OUR selects
        while sess.obs["current"]["yourIndex"] != me:
            sess.select(random_picks(sess.obs, rng))
            if sess.done:
                raise AssertionError("game ended before a probe point")
        tracker.update(sess.obs.get("logs") or [])
        obs = sess.obs
        belief = tracker.snapshot()

        n_rows = tables.n_rows
        r = np.random.RandomState(0)
        dl = np.abs(r.normal(0.05, 0.05, n_rows)).astype(np.float32)
        for cid in deck:                       # informed-ish decklist prior
            dl[cid + 2] += 1.0
        hd = np.abs(r.normal(0.02, 0.02, n_rows)).astype(np.float32)

        ss = SearchSession()
        srng = random.Random(5)
        accepted = 0
        for i in range(50):
            det = sample_determinization(obs, me, deck, belief, dl, hd,
                                         tables, srng)
            _counts_ok(det, obs, me)
            # tracker known-hand cards are a hard minimum of the sampled hand
            hand_c = Counter(det.opp_hand)
            for cid, n in belief.opp_hand.items():
                assert hand_c[cid] >= n, (cid, n, hand_c)
            # revealed opp prizes preserved at their indices
            for j, c in enumerate(obs["current"]["players"][1 - me]["prize"] or []):
                if c is not None:
                    assert det.opp_prize[j] == c["id"]
            if ss.begin(obs, det) is not None:
                accepted += 1
        ss.end()
        assert accepted == 50, f"engine rejected {50 - accepted}/50 samples"

        # filler fallback: exact counts, engine-accepted, no aux/belief needed
        det = filler_determinization(obs, me, deck, tables, srng)
        _counts_ok(det, obs, me)
        got = ss.begin(obs, det)
        assert got is not None
        ss.end()
    finally:
        sess.close()


def test_never_raises_on_garbage_obs():
    tables = build_tables()
    rng = random.Random(0)
    for obs in ({}, {"current": None}, {"current": {"players": []}},
                {"current": {"players": [{}, {}], "yourIndex": 0},
                 "select": {}}):
        d = sample_determinization(obs, 0, [3] * 60, None, None, None,
                                   tables, rng)
        assert isinstance(d, Determinization)
        d2 = filler_determinization(obs, 0, [3] * 60, tables, rng)
        assert isinstance(d2, Determinization)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_determinize.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'ptcg.determinize'`

- [ ] **Step 3: Write the implementation**

Create `ptcg/determinize.py`:

```python
"""Sample engine-ready hidden-info assignments (determinizations).

Opponent side merges three information layers:
  1. visible cards (discard, board incl. attachments/pre-evolutions,
     revealed prizes, their stadium) are fixed and excluded;
  2. BeliefTracker constraints: known hand/deck membership are hard
     minimums, the hidden pool must land in hand+deck;
  3. aux heads: dl_lam (expected full-decklist counts per card row)
     weights unknown-card sampling, hd_lam weights which unknowns sit in
     the hand. A 4-per-card-id cap (basic energy exempt) and a 1-ACE-SPEC
     cap keep samples realistic (approximates the 4-per-name rule; the
     inference tables carry no names).
Own side: unseen = decklist - hand - visible; the deck/prize split is
sampled uniformly (which cards are prized IS uniform; ordering is
engine-shuffled regardless of what we supply).
Contract: never raises; exact zone counts whenever the obs is well-formed.
"""
import random
from collections import Counter
from dataclasses import dataclass

from .cards import (A_ACESPEC, A_BASIC, A_IS_BASIC_ENERGY, A_IS_POKEMON,
                    N_RESERVED)

_BASIC_ENERGY_IDS = (1, 2, 3, 4, 5, 6, 7, 8)   # card id == EnergyType 1..8


@dataclass(frozen=True)
class Determinization:
    your_deck: list
    your_prize: list
    opp_deck: list
    opp_prize: list
    opp_hand: list
    opp_active: list
    opp_decklist: list      # full 60 for featurizing opponent nodes


def _board_ids(pstate):
    out = []
    for pk in (pstate.get("active") or []) + (pstate.get("bench") or []):
        if not pk:
            continue
        out.append(pk["id"])
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in pk.get(grp) or []:
                if c:
                    out.append(c["id"])
    return out


def _visible_counts(obs, pidx):
    p = obs["current"]["players"][pidx]
    ids = [c["id"] for c in p.get("discard") or [] if c]
    ids += _board_ids(p)
    ids += [c["id"] for c in p.get("prize") or [] if c]     # revealed prizes
    for c in obs["current"].get("stadium") or []:
        if c and c.get("playerIndex") == pidx:
            ids.append(c["id"])
    return Counter(ids)


def _energy_filler(obs, opp):
    """A basic-energy id matching the opponent's dominant board energy."""
    try:
        p = obs["current"]["players"][opp]
        et = Counter()
        for pk in (p.get("active") or []) + (p.get("bench") or []):
            if pk:
                for e in pk.get("energies") or []:
                    et[int(e)] += 1
        for e, _ in et.most_common():
            if 1 <= e <= 8:
                return e
    except Exception:
        pass
    return 3                                                # water energy


def _fallback_basic(tables):
    """Lowest-id Basic Pokemon in the pool (a valid, safe active guess)."""
    attr = tables.attr
    for row in range(N_RESERVED, tables.n_rows):
        if attr[row, A_IS_POKEMON] and attr[row, A_BASIC]:
            return row - N_RESERVED
    return _BASIC_ENERGY_IDS[2]     # unreachable with real tables


def _sample_counts(weights, k, rng, assigned, attr, n_rows):
    """k card ids by weighted draw; count-decrement without replacement.

    weights: dict row -> float. assigned: Counter id -> already-placed count
    (visible + known + sampled so far), used for the 4-per-id cap (basic
    energy exempt) and the 1-ACE-SPEC cap.
    """
    out = []
    w = {r: float(v) for r, v in weights.items() if v > 0}
    ace_used = any(attr[cid + N_RESERVED, A_ACESPEC]
                   for cid in assigned if 0 <= cid + N_RESERVED < n_rows)
    for _ in range(k):
        live = []
        tot = 0.0
        for r, v in w.items():
            if v <= 0:
                continue
            cid = r - N_RESERVED
            if not attr[r, A_IS_BASIC_ENERGY] and assigned[cid] >= 4:
                continue
            if ace_used and attr[r, A_ACESPEC]:
                continue
            live.append((r, v))
            tot += v
        if tot <= 0 or not live:
            break
        x = rng.random() * tot
        acc = 0.0
        for r, v in live:
            acc += v
            if acc >= x:
                cid = r - N_RESERVED
                out.append(cid)
                assigned[cid] += 1
                w[r] = v - 1.0
                if attr[r, A_ACESPEC]:
                    ace_used = True
                break
    return out


def _pick_indices(weights, k, rng):
    """k distinct indices, probability proportional to weight (removal)."""
    w = {i: float(v) for i, v in enumerate(weights) if v > 0}
    out = []
    for _ in range(min(k, len(w))):
        tot = sum(w.values())
        if tot <= 0:
            out.extend(list(w.keys())[:k - len(out)])
            break
        x = rng.random() * tot
        acc = 0.0
        for i, v in list(w.items()):
            acc += v
            if acc >= x:
                out.append(i)
                del w[i]
                break
    return out


def sample_determinization(obs, me, my_deck, belief, dl_lam, hd_lam,
                           tables, rng):
    try:
        return _sample(obs, me, my_deck, belief, dl_lam, hd_lam, tables, rng)
    except Exception:
        return filler_determinization(obs, me, my_deck, tables, rng)


def filler_determinization(obs, me, my_deck, tables, rng):
    """Constraint-free fallback: no belief, no aux guidance."""
    try:
        return _sample(obs, me, my_deck, None, None, None, tables, rng)
    except Exception:
        return Determinization([], [], [], [], [], [], [])


def _sample(obs, me, my_deck, belief, dl_lam, hd_lam, tables, rng):
    cur = obs["current"]
    opp = 1 - me
    pme, popp = cur["players"][me], cur["players"][opp]
    attr, n_rows = tables.attr, tables.n_rows
    filler = _energy_filler(obs, opp)

    # ---------- my side ----------
    unseen = Counter(int(c) for c in my_deck)
    unseen.subtract(Counter(c["id"] for c in pme.get("hand") or [] if c))
    unseen.subtract(_visible_counts(obs, me))
    for c in cur.get("looking") or []:
        if c and c.get("playerIndex") == me:
            unseen[c["id"]] -= 1
    pool = [cid for cid, n in unseen.items() for _ in range(max(n, 0))]
    rng.shuffle(pool)
    n_deck = int(pme["deckCount"])
    my_prize = list(pme.get("prize") or [])
    n_unrev = sum(1 for c in my_prize if c is None)
    while len(pool) < n_deck + n_unrev:
        pool.append(int(rng.choice(my_deck)))
    your_deck = pool[:n_deck]
    it = iter(pool[n_deck:n_deck + n_unrev])
    your_prize = [(c["id"] if c else next(it)) for c in my_prize]

    # ---------- opponent side ----------
    vis = _visible_counts(obs, opp)
    kh = Counter(dict(belief.opp_hand)) if belief else Counter()
    kd = Counter(dict(belief.opp_deck)) if belief else Counter()
    kp = Counter(dict(belief.opp_hidden_pool)) if belief else Counter()
    n_hand = int(popp["handCount"])
    n_deck_o = int(popp["deckCount"])
    opp_prize_l = list(popp.get("prize") or [])
    n_unrev_o = sum(1 for c in opp_prize_l if c is None)
    hidden_total = n_hand + n_deck_o + n_unrev_o
    known = [cid for cnt in (kh, kd, kp) for cid, n in cnt.items()
             for _ in range(n)]
    n_unknown = max(0, hidden_total - len(known))

    assigned = Counter(vis)
    for cid in known:
        assigned[cid] += 1
    weights = {}
    for row in range(N_RESERVED, n_rows):
        if not attr[row].any():
            continue
        base = float(dl_lam[row]) if dl_lam is not None else 0.0
        w = base - assigned[row - N_RESERVED]   # predicted total minus placed
        if w > 0:
            weights[row] = w
    unknown = _sample_counts(weights, n_unknown, rng, assigned, attr, n_rows)
    while len(unknown) < n_unknown:
        unknown.append(filler)

    # zone assignment: known hand first, hd_lam weights the rest of the hand
    hand = [cid for cid, n in kh.items() for _ in range(n)][:n_hand]
    cand = [cid for cid, n in kp.items() for _ in range(n)] + unknown
    need = n_hand - len(hand)
    if need > 0:
        hw = [(float(hd_lam[c + N_RESERVED]) if hd_lam is not None
               and 0 <= c + N_RESERVED < n_rows else 0.0) + 0.05
              for c in cand]
        idx = set(_pick_indices(hw, need, rng))
        hand += [cand[i] for i in sorted(idx)]
        cand = [cand[i] for i in range(len(cand)) if i not in idx]
    while len(hand) < n_hand:
        hand.append(filler)
    hand = hand[:n_hand]

    known_deck_list = [cid for cid, n in kd.items() for _ in range(n)]
    rng.shuffle(cand)
    used_for_deck = max(0, n_deck_o - len(known_deck_list))
    deck = known_deck_list + cand[:used_for_deck]
    rest = cand[used_for_deck:]
    while len(deck) < n_deck_o:
        deck.append(filler)
    deck = deck[:n_deck_o]

    prize_pool = rest
    while len(prize_pool) < n_unrev_o:
        prize_pool.append(filler)
    it2 = iter(prize_pool)
    opp_prize = [(c["id"] if c else next(it2)) for c in opp_prize_l]

    # facedown active + setup basic requirements
    opp_active = []
    oa = popp.get("active") or []
    if oa and oa[0] is None:
        best_row, best_w = None, -1.0
        for row, w in weights.items():
            if attr[row, A_IS_POKEMON] and attr[row, A_BASIC] and w > best_w:
                best_row, best_w = row, w
        opp_active = [best_row - N_RESERVED if best_row is not None
                      else _fallback_basic(tables)]
    if int(cur.get("turn") or 0) == 0:
        has_basic = any(
            0 <= c + N_RESERVED < n_rows
            and attr[c + N_RESERVED, A_IS_POKEMON]
            and attr[c + N_RESERVED, A_BASIC] for c in deck)
        if not has_basic and deck:
            deck[-1] = _fallback_basic(tables)

    decklist = ([c for c in vis.elements()] + hand + deck
                + [c for c in opp_prize])
    while len(decklist) < 60:
        decklist.append(filler)
    decklist = decklist[:60]

    return Determinization(
        [int(x) for x in your_deck], [int(x) for x in your_prize],
        [int(x) for x in deck], [int(x) for x in opp_prize],
        [int(x) for x in hand], [int(x) for x in opp_active],
        [int(x) for x in decklist])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_determinize.py -v`
Expected: 2 passed (the 50-sample engine-acceptance loop takes a few seconds)

- [ ] **Step 5: Commit**

```bash
git add ptcg/determinize.py tests/test_determinize.py
git commit -m "add determinizer: belief+aux-guided hidden-info sampling"
```

---

### Task 4: `ptcg/mcts.py` core — action proposal, PUCT node math (fake-engine units)

**Files:**
- Create: `ptcg/mcts.py`
- Test: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `ptcg.action.run_pick_loop(model, trunk, sb, selb, *, forced=None, generator=None) -> (picks, logp, ent)`; `ptcg.model.collate_states/collate_selects`; `model.encode/option_logits/public_value/aux_decklist/aux_hand`; `ptcg.featurize.featurize_state(obs, me, own_deck, belief, tables)` / `encode_select(obs, ts, tables)`; `ptcg.tracker.BeliefTracker` (attrs `me/_hand/_deck/_pool`); `SearchSession` (Task 2); `sample_determinization`/`filler_determinization` (Task 3).
- Produces: `@dataclass SearchConfig(k_trees=6, sims_per_tree=64, c_puct=1.5, m_multipick=8)`; `@dataclass MoveStats(searched=False, trees=0, sims=0, elapsed=0.0, reason="")`; `propose_actions(model, ts, es, select, gen, m_multipick=8) -> (actions: list[tuple], priors: list[float], trunk)`; `search_move(obs, me, my_deck, tracker, model, tables, session, cfg, rng, gen, tslice) -> (list[int] | None, MoveStats)`; internals `_Node`, `_select_action(node, c_puct)`, `_simulate(...)`, `_vote(roots) -> tuple | None`, `_clone(tracker)`, `_child_trackers(pair, child_obs)`.

- [ ] **Step 1: Write the failing unit tests (fakes — no engine, no torch model)**

Create `tests/test_mcts.py`:

```python
import math

import ptcg.mcts as M


class _FakeSession:
    """Scripted 2-ply game. sid 0 = root (my seat 0).
    action (0,) -> sid 1: terminal, I win.
    action (1,) -> sid 2: opponent node (seat 1) with 2 actions."""

    def __init__(self):
        self.tree = {
            (0, (0,)): (1, _obs(seat=0, result=0)),           # I win
            (0, (1,)): (2, _obs(seat=1, result=-1)),          # opp to move
            (2, (0,)): (3, _obs(seat=0, result=1)),           # opp wins
            (2, (1,)): (4, _obs(seat=0, result=-1)),          # play on
        }

    def step(self, sid, picks):
        return self.tree.get((sid, tuple(picks)))

    def end(self):
        pass


def _obs(seat, result):
    return {"current": {"yourIndex": seat, "result": result,
                        "players": [{}, {}]},
            "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
            "logs": []}


class _FakeTracker:
    def __init__(self, me):
        self.me = me
        self._hand = {}
        self._deck = {}
        self._pool = {}

    def update(self, logs):
        pass

    def snapshot(self):
        return None


def _mk_root(session):
    root = M._Node(0, _obs(seat=0, result=-1), 0,
                   (_FakeTracker(0), _FakeTracker(1)))
    root.actions = [(0,), (1,)]
    root.P = [0.5, 0.5]
    root.N = [0, 0]
    root.W = [0.0, 0.0]
    return root


def test_select_action_puct_math():
    root = _mk_root(_FakeSession())
    root.N = [3, 1]
    root.W = [1.5, 0.9]
    c = 1.5
    tot = math.sqrt(4 + 1)
    s0 = 0.5 + c * 0.5 * tot / 4
    s1 = 0.9 + c * 0.5 * tot / 2
    want = 0 if s0 >= s1 else 1
    assert M._select_action(root, c) == want


def test_simulate_backs_up_negamax(monkeypatch):
    # opponent leaf evaluates +0.9 FOR THE OPPONENT -> -0.9 for me at root
    def fake_eval(model, obs, seat, deck, belief, tables, gen, m):
        return [(0,), (1,)], [0.5, 0.5], 0.9
    monkeypatch.setattr(M, "_eval_state", fake_eval)
    sess = _FakeSession()
    root = _mk_root(sess)
    # force the (1,) branch: bias priors
    root.P = [0.0, 1.0]
    ran = M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                      M.SearchConfig())
    assert ran
    assert root.N == [0, 1]
    assert abs(root.W[1] - (-0.9)) < 1e-9      # flipped into my perspective


def test_simulate_terminal_and_dead_edge(monkeypatch):
    monkeypatch.setattr(M, "_eval_state",
                        lambda *a: ([(0,)], [1.0], 0.0))
    sess = _FakeSession()
    root = _mk_root(sess)
    root.P = [1.0, 0.0]                         # go to the winning terminal
    M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                M.SearchConfig())
    assert root.N[0] == 1 and abs(root.W[0] - 1.0) < 1e-9
    # engine refusing a step becomes a neutral dead edge, not a crash
    root2 = _mk_root(sess)
    root2.actions = [(9,), (0,)]                # (9,) unknown to the fake
    root2.P = [1.0, 0.0]
    root2.N = [0, 0]
    root2.W = [0.0, 0.0]
    M._simulate(root2, 0, [3] * 60, [3] * 60, None, None, sess, None,
                M.SearchConfig())
    assert root2.N[0] == 1 and root2.W[0] == 0.0


def test_vote_sums_across_trees_and_breaks_ties_by_value():
    sess = _FakeSession()
    a = _mk_root(sess)
    b = _mk_root(sess)
    a.N, a.W = [3, 1], [1.0, 0.5]
    b.N, b.W = [1, 3], [0.2, 0.4]
    assert M._vote([(a, None), (b, None)]) in ((0,), (1,))
    b.N = [1, 5]                                # (1,) now dominates 4 vs 6
    assert M._vote([(a, None), (b, None)]) == (1,)
    # tie on visits -> higher mean value wins
    c1, c2 = _mk_root(sess), _mk_root(sess)
    c1.N, c1.W = [2, 2], [1.8, 0.2]
    assert M._vote([(c1, None)]) == (0,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcts.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'ptcg.mcts'`

- [ ] **Step 3: Write the implementation**

Create `ptcg/mcts.py`:

```python
"""K-tree PUCT search over the engine's search API.

One PUCT tree per determinization; root actions/priors are shared (the
root select is the real one in every tree). Values are stored from each
node's acting seat and negamax-flipped on backup; leaves are evaluated by
the policy's public_value head from the acting seat's perspective, with
the same information structure the net trained on (that seat's belief
tracker + decklist). Chance (draws/shuffles/coins) stays implicit: each
expansion samples one outcome and caches it (determinized-UCT bias,
mitigated by the K independent trees). The root decision is the argmax of
summed visit counts across trees over exact pick tuples (order-preserving,
safe for order-sensitive contexts, at worst splitting equivalent votes).
"""
import math
import time
from collections import Counter
from dataclasses import dataclass

from .tracker import BeliefTracker


@dataclass
class SearchConfig:
    k_trees: int = 6
    sims_per_tree: int = 64
    c_puct: float = 1.5
    m_multipick: int = 8


@dataclass
class MoveStats:
    searched: bool = False
    trees: int = 0
    sims: int = 0
    elapsed: float = 0.0
    reason: str = ""


class _Node:
    __slots__ = ("sid", "obs", "seat", "trk", "term_v",
                 "actions", "P", "N", "W", "children")

    def __init__(self, sid, obs, seat, trk, term_v=None):
        self.sid = sid
        self.obs = obs
        self.seat = seat
        self.trk = trk              # (tracker seat 0, tracker seat 1)
        self.term_v = term_v        # value for the ROOT player if terminal
        self.actions = None         # None until evaluated (leaf)
        self.P = self.N = self.W = None
        self.children = {}


def _clone(t):
    n = BeliefTracker(t.me)
    n._hand = Counter(t._hand)
    n._deck = Counter(t._deck)
    n._pool = Counter(t._pool)
    return n


def _child_trackers(pair, child_obs):
    s = child_obs["current"]["yourIndex"]
    t = _clone(pair[s])
    t.update(child_obs.get("logs") or [])
    out = list(pair)
    out[s] = t
    return tuple(out)


def _greedy_picks(model, trunk, sb, selb):
    import torch
    O = selb["opt_type"].shape[1]
    max_count = int(selb["max_count_t"][0])
    picked = torch.zeros((1, O + 1), dtype=torch.bool)
    picks = []
    while True:
        logits = model.option_logits(trunk, sb, selb, picked)
        a = int(logits.argmax(dim=-1))
        if a == O:
            break
        picks.append(a)
        picked = picked.clone()
        picked[0, a] = True
        if len(picks) == max_count:
            break
    return picks


def propose_actions(model, ts, es, select, gen, m_multipick=8):
    """Candidate pick-lists + priors for one select.

    Returns (actions: list[tuple[int,...]], priors: list[float], trunk).
    Single-pick selects (98% of nodes) cost one option_logits pass; the
    empty decline () is a candidate iff minCount==0 (prior = done column).
    Multi-pick selects use m sampled pick-lists plus the greedy one.
    """
    import torch

    from .action import run_pick_loop
    from .model import collate_selects, collate_states

    sb = collate_states([ts])
    selb = collate_selects([es])
    n = len(select["option"])
    lo, hi = int(select["minCount"]), int(select["maxCount"])
    with torch.no_grad():
        trunk = model.encode(sb)
        if hi == 1:
            O = selb["opt_type"].shape[1]
            picked = torch.zeros((1, O + 1), dtype=torch.bool)
            logits = model.option_logits(trunk, sb, selb, picked)[0]
            probs = torch.softmax(logits, dim=-1)
            actions = [(j,) for j in range(n)]
            pri = [float(probs[j]) for j in range(n)]
            if lo == 0:
                actions.append(())
                pri.append(float(probs[O]))
        else:
            cand = {}
            for _ in range(m_multipick):
                picks, logp, _ = run_pick_loop(model, trunk, sb, selb,
                                               generator=gen)
                key = tuple(picks)
                cand[key] = max(cand.get(key, 0.0),
                                math.exp(min(float(logp), 0.0)))
            g = tuple(_greedy_picks(model, trunk, sb, selb))
            if g not in cand:
                cand[g] = max(cand.values()) if cand else 1.0
            actions = list(cand.keys())
            pri = [cand[a] for a in actions]
    tot = sum(pri) or 1.0
    return actions, [p / tot for p in pri], trunk


def _eval_state(model, obs, seat, deck, belief, tables, gen, m_multipick):
    """Featurize a search obs from its acting seat; return actions, priors,
    and public_value FROM THAT SEAT's perspective."""
    import torch

    from .featurize import encode_select, featurize_state
    ts = featurize_state(obs, seat, deck, belief, tables)
    es = encode_select(obs, ts, tables)
    actions, priors, trunk = propose_actions(model, ts, es, obs["select"],
                                             gen, m_multipick)
    with torch.no_grad():
        v = float(model.public_value(trunk))
    return actions, priors, v


def _select_action(node, c_puct):
    sqrt_total = math.sqrt(sum(node.N) + 1)
    best_i, best_s = 0, float("-inf")
    for i in range(len(node.actions)):
        q = node.W[i] / node.N[i] if node.N[i] else 0.0
        s = q + c_puct * node.P[i] * sqrt_total / (1 + node.N[i])
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def _simulate(root, me, my_deck, opp_decklist, model, tables, session, gen,
              cfg):
    """One PUCT simulation: descend, expand/evaluate one leaf, back up."""
    path = []
    node = root
    while True:
        if node.term_v is not None:
            v_me = node.term_v
            break
        if node.actions is None:            # unexpanded: evaluate + stop
            seat = node.seat
            deck = my_deck if seat == me else opp_decklist
            belief = node.trk[seat].snapshot()
            try:
                actions, priors, v = _eval_state(
                    model, node.obs, seat, deck, belief, tables, gen,
                    cfg.m_multipick)
            except Exception:
                actions = []
            if not actions:
                node.term_v = 0.0
                v_me = 0.0
                break
            node.actions = actions
            node.P = priors
            node.N = [0] * len(actions)
            node.W = [0.0] * len(actions)
            v_me = v if seat == me else -v
            break
        a = _select_action(node, cfg.c_puct)
        path.append((node, a))
        child = node.children.get(a)
        if child is None:
            nxt = session.step(node.sid, list(node.actions[a]))
            if nxt is None:                 # engine refused: neutral dead edge
                child = _Node(-1, None, node.seat, node.trk, term_v=0.0)
            else:
                sid, obs = nxt
                res = obs["current"]["result"]
                if res != -1:
                    tv = 1.0 if res == me else (0.0 if res == 2 else -1.0)
                    child = _Node(sid, obs, obs["current"]["yourIndex"],
                                  node.trk, term_v=tv)
                else:
                    child = _Node(sid, obs, obs["current"]["yourIndex"],
                                  _child_trackers(node.trk, obs))
            node.children[a] = child
        node = child
    for n, a in path:
        n.N[a] += 1
        n.W[a] += v_me if n.seat == me else -v_me
    return True


def _vote(roots):
    """Root pick across trees: max summed visits, ties by mean value."""
    votes, wsum = Counter(), Counter()
    for root, _ in roots:
        if not root.actions:
            continue
        for j, a in enumerate(root.actions):
            votes[a] += root.N[j]
            wsum[a] += root.W[j]
    if not votes:
        return None
    return max(votes, key=lambda a: (votes[a],
                                     wsum[a] / votes[a] if votes[a] else float("-inf")))


def search_move(obs, me, my_deck, tracker, model, tables, session, cfg, rng,
                gen, tslice):
    """Search-chosen pick list for the agent's current obs, or None.

    None means "no answer" (caller falls back to raw policy). Never raises;
    always search_end()s the arena before returning.
    """
    import torch
    t0 = time.perf_counter()
    stats = MoveStats()
    try:
        from .determinize import (filler_determinization,
                                  sample_determinization)
        from .featurize import encode_select, featurize_state
        select = obs["select"]
        belief = tracker.snapshot()
        ts = featurize_state(obs, me, my_deck, belief, tables)
        es = encode_select(obs, ts, tables)
        actions, priors, trunk = propose_actions(model, ts, es, select, gen,
                                                 cfg.m_multipick)
        if len(actions) <= 1:
            stats.reason = "single-action"
            return (list(actions[0]) if actions else None), stats
        with torch.no_grad():
            dl = model.aux_decklist(trunk)[0].numpy()
            hd = model.aux_hand(trunk)[0].numpy()
        roots = []
        for _ in range(cfg.k_trees):
            det = sample_determinization(obs, me, my_deck, belief, dl, hd,
                                         tables, rng)
            got = session.begin(obs, det)
            if got is None:
                det = filler_determinization(obs, me, my_deck, tables, rng)
                got = session.begin(obs, det)
            if got is None:
                continue
            sid, robs = got
            trk_me, trk_opp = _clone(tracker), BeliefTracker(1 - me)
            pair = (trk_me, trk_opp) if me == 0 else (trk_opp, trk_me)
            root = _Node(sid, robs, me, pair)
            root.actions = list(actions)
            root.P = list(priors)
            root.N = [0] * len(actions)
            root.W = [0.0] * len(actions)
            roots.append((root, det.opp_decklist))
        stats.trees = len(roots)
        if not roots:
            stats.reason = "no-roots"
            return None, stats
        max_sims = cfg.k_trees * cfg.sims_per_tree
        i = 0
        while (time.perf_counter() - t0) < tslice and stats.sims < max_sims:
            root, odl = roots[i % len(roots)]
            _simulate(root, me, my_deck, odl, model, tables, session, gen,
                      cfg)
            stats.sims += 1
            i += 1
        if stats.sims == 0:
            stats.reason = "no-sims"
            return None, stats
        best = _vote(roots)
        if best is None:
            stats.reason = "no-vote"
            return None, stats
        stats.searched = True
        return list(best), stats
    except Exception:
        stats.reason = "error"
        return None, stats
    finally:
        stats.elapsed = time.perf_counter() - t0
        try:
            session.end()
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcts.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ptcg/mcts.py tests/test_mcts.py
git commit -m "add mcts core: puct node math, negamax backup, root vote"
```

---

### Task 5: `search_move` end-to-end on the live engine + real model

**Files:**
- Modify: `tests/test_mcts.py` (append tests; `ptcg/mcts.py` only if a defect surfaces)

**Interfaces:**
- Consumes: everything from Tasks 2–4; `submission_src/policy.pt` (local, gitignored); `ptcg.model.PolicyModel, student_config`.
- Produces: proven `search_move` behavior later tasks rely on: legal picks, `stats.searched/sims/trees/elapsed`, wall-clock compliance, single-action shortcut.

- [ ] **Step 1: Append the failing integration test**

Append to `tests/test_mcts.py`:

```python
import os
import random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_search_move_live_engine_legal_and_budgeted():
    import torch

    from ptcg.cards import build_tables
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.model import PolicyModel, student_config
    from ptcg.simsearch import SearchSession
    from ptcg.tracker import BeliefTracker

    tables = build_tables()
    model = PolicyModel(student_config(tables))
    model.load_state_dict(torch.load(
        os.path.join(_REPO, "submission_src", "policy.pt"),
        map_location="cpu", weights_only=True))
    model.eval()

    deck = load_sample_deck()
    sess = BattleSession(deck, deck)
    rng = random.Random(21)
    try:
        me = sess.obs["current"]["yourIndex"]
        tracker = BeliefTracker(me)
        for _ in range(20):
            if sess.done:
                break
            if sess.obs["current"]["yourIndex"] == me:
                tracker.update(sess.obs.get("logs") or [])
            sess.select(random_picks(sess.obs, rng))
        while sess.obs["current"]["yourIndex"] != me and not sess.done:
            sess.select(random_picks(sess.obs, rng))
        assert not sess.done
        tracker.update(sess.obs.get("logs") or [])
        obs = sess.obs

        ss = SearchSession()
        cfg = M.SearchConfig(k_trees=2, sims_per_tree=8)
        gen = torch.Generator().manual_seed(0)
        t0 = __import__("time").perf_counter()
        picks, stats = M.search_move(obs, me, deck, tracker, model, tables,
                                     ss, cfg, random.Random(1), gen,
                                     tslice=4.0)
        dt = __import__("time").perf_counter() - t0
        sel = obs["select"]
        if stats.searched:
            assert isinstance(picks, list)
            assert sel["minCount"] <= len(picks) <= sel["maxCount"]
            assert len(set(picks)) == len(picks)
            assert all(0 <= p < len(sel["option"]) for p in picks)
            assert stats.sims >= 1 and stats.trees >= 1
        else:
            # single-action selects shortcut without searching
            assert stats.reason == "single-action" and picks is not None
        # slice + one leaf-eval of overshoot is the budget contract
        assert dt < 4.0 + 2.5, f"took {dt:.1f}s"
        assert stats.elapsed <= dt
    finally:
        sess.close()
```

- [ ] **Step 2: Run the new test to verify current state**

Run: `python -m pytest tests/test_mcts.py -v -k live_engine`
Expected: PASS if Tasks 2–4 are correct — this step is a genuine integration
gate, not a formality: if it FAILS, debug `mcts.py`/`simsearch.py` (typical
causes: featurizing a search obs whose acting seat differs from `me`;
tracker pair ordering; stepping with a tuple instead of list) and fix before
committing. Also re-run the full file: `python -m pytest tests/test_mcts.py -v`
Expected: 5 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts.py ptcg/mcts.py
git commit -m "prove search_move end-to-end on the live engine"
```

---

### Task 6: agent integration in `submission_src/main.py`

**Files:**
- Modify: `submission_src/main.py` (full new content below)
- Test: `tests/test_submission_agent.py` (append tests)

**Interfaces:**
- Consumes: `ptcg.clock.SearchClock/forced_picks`, `ptcg.simsearch.SearchSession`, `ptcg.mcts.SearchConfig/search_move` (exact signatures from Tasks 1/2/4).
- Produces (used by Tasks 7–9): module globals `_SEARCH_ENABLED` (bool literal on its own line, flipped by the `--no-search` build), `_TELEM` dict with keys `games, moves, searched, sims, fallbacks, search_time`; helpers `_configure_search(bank_s=None, floor_s=None, cap_s=None, k_trees=None, sims_per_tree=None)` and `_reseed(seed: int)`; unchanged public surface `agent(obs_dict)`, `_is_legal(picks, sel)`, `_fallback(obs_dict)`, `_ensure_model()`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_submission_agent.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_submission_agent.py -v`
Expected: the 4 new tests FAIL (`AttributeError: ... has no attribute '_SEARCH_ENABLED'` etc.); the 5 existing tests still PASS.

- [ ] **Step 3: Replace `submission_src/main.py` with the integrated agent**

Full new content of `submission_src/main.py`:

```python
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
```

Notes for the implementer (behavioral invariants to keep):
- The trivial fast path runs before any torch import — `_MODEL` stays None if only trivial selects arrive (tested).
- The tracker update happens exactly once per call, before any pick path (unchanged from the current agent).
- When `_SEARCH_ENABLED` is False and no search modules are importable, behavior is exactly the pre-phase-4 agent (plus the trivial fast path and telemetry counters).
- The double `note_move` guard: `note_move` is called when `_CLOCK` exists; the `if _CLOCK is None and _SEARCH_ENABLED:` block right after `_ensure_model()` creates the clock on the first non-trivial move and immediately notes that move so it isn't lost. Do not call `note_move` twice for one obs — follow the code above exactly.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_submission_agent.py -v`
Expected: 9 passed (5 existing + 4 new)

- [ ] **Step 5: Run the existing full-game smoke to prove no regression**

Run: `PYTHONIOENCODING=utf-8 python scripts/test_submission.py 2`
Expected: `games=2 ... OK — all agent picks legal, all games completed`, win rate 1.0 vs random. (Search runs at default budgets here — expect a few seconds per searched move, total a few minutes; if it exceeds ~15 min something is wrong with the slice logic.)

- [ ] **Step 6: Commit**

```bash
git add submission_src/main.py tests/test_submission_agent.py
git commit -m "integrate search into the submission agent behind the fallback ladder"
```

---

### Task 7: packaging — bundle the new modules, `--no-search` rollback build

**Files:**
- Modify: `scripts/make_submission.py`

**Interfaces:**
- Consumes: `submission_src/main.py`'s `_SEARCH_ENABLED = True` literal line (flipped textually for the rollback build); `ptcg.simsearch.SearchSession.ensure_ptr()`.
- Produces: `dist/submission/` + `dist/submission.zip` (search build, default); `dist/submission-nosearch/` + `dist/submission-nosearch.zip` (rollback) when run with `--no-search`.

- [ ] **Step 1: Extend the build script**

Replace `scripts/make_submission.py` content with:

```python
"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python).
Usage: python scripts/make_submission.py [--no-search]
--no-search builds dist/submission-nosearch/ with _SEARCH_ENABLED flipped to
False in main.py — a rollback bundle behaviorally identical to the validated
pre-search agent."""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
CG = os.path.join(REPO, "pokemon-tcg-ai-battle", "sample_submission",
                  "sample_submission", "cg")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py",
                "clock.py", "simsearch.py", "determinize.py", "mcts.py"]


def main(no_search=False):
    name = "submission-nosearch" if no_search else "submission"
    out = os.path.join(REPO, "dist", name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in ("deck.csv", "policy.pt", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(out, f))
    main_src = open(os.path.join(SRC, "main.py")).read()
    if no_search:
        flipped = main_src.replace("_SEARCH_ENABLED = True",
                                   "_SEARCH_ENABLED = False", 1)
        assert flipped != main_src, "_SEARCH_ENABLED literal not found"
        main_src = flipped
    with open(os.path.join(out, "main.py"), "w") as f:
        f.write(main_src)
    os.makedirs(os.path.join(out, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(out, "ptcg", m))
    shutil.copytree(CG, os.path.join(out, "cg"))
    # Self-containment check: from a subprocess with cwd=out and the REPO
    # stripped from sys.path, the bundle must exec main.py the Kaggle way
    # (no __file__), lazily load the native model + bundled cg engine,
    # return a legal deck, and (search build) create a search arena.
    search_check = (
        "import ptcg.simsearch as ss\n"
        "assert ss.SearchSession().ensure_ptr(), 'AgentStart failed'\n"
        "print('search arena OK')\n"
    ) if not no_search else ""
    check = (
        "import os, sys\n"
        f"repo = {REPO!r}\n"
        "sys.path = [p for p in sys.path if os.path.abspath(p or '.') != repo]\n"
        "sys.path.insert(0, os.getcwd())\n"
        "ns = {}\n"
        "exec(compile(open('main.py').read(), 'main.py', 'exec'), ns)\n"
        f"assert ns['_SEARCH_ENABLED'] is {not no_search}\n"
        "ns['_ensure_model']()\n"
        "assert ns['_MODEL'] is not None\n"
        "d = ns['agent']({'select': None, 'current': None, 'logs': []})\n"
        "assert len(d) == 60\n"
        + search_check +
        "print('self-contained OK: no __file__, model+cg from bundle, deck', len(d))\n"
    )
    r = subprocess.run([sys.executable, "-c", check], cwd=out,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    print(r.stderr.strip())
    if r.returncode != 0:
        sys.exit("self-containment check FAILED")
    for root, dirs, _ in os.walk(out):
        for dname in list(dirs):
            if dname == "__pycache__":
                shutil.rmtree(os.path.join(root, dname))
    zip_base = os.path.join(REPO, "dist", name)
    shutil.make_archive(zip_base, "zip", out)
    print(f"wrote {out}/ and {zip_base}.zip")


if __name__ == "__main__":
    main(no_search="--no-search" in sys.argv[1:])
```

- [ ] **Step 2: Build both bundles and verify**

Run: `PYTHONIOENCODING=utf-8 python scripts/make_submission.py`
Expected output includes: `search arena OK`, `self-contained OK: no __file__, model+cg from bundle, deck 60`, `wrote ...dist/submission/ and ...dist/submission.zip`

Run: `PYTHONIOENCODING=utf-8 python scripts/make_submission.py --no-search`
Expected output includes: `self-contained OK ... deck 60` (NO `search arena OK` line), `wrote ...dist/submission-nosearch/ and ...dist/submission-nosearch.zip`

- [ ] **Step 3: Verify the rollback flip landed in the bundle**

Run: `grep -n "_SEARCH_ENABLED" dist/submission-nosearch/main.py dist/submission/main.py`
Expected: `False` in submission-nosearch/main.py, `True` in submission/main.py

- [ ] **Step 4: Commit**

```bash
git add scripts/make_submission.py
git commit -m "bundle search modules and add --no-search rollback build"
```

---

### Task 8: forced-search game smoke in `scripts/test_submission.py`

**Files:**
- Modify: `scripts/test_submission.py`

**Interfaces:**
- Consumes: `mod._configure_search(...)`, `mod._TELEM`, `mod._reseed(seed)` from Task 6.
- Produces: `python scripts/test_submission.py <n> --small-search` mode used by Task 9's acceptance run.

- [ ] **Step 1: Extend the smoke**

In `scripts/test_submission.py`, replace the `main()` function with:

```python
def main():
    args = [a for a in sys.argv[1:]]
    small_search = "--small-search" in args
    args = [a for a in args if not a.startswith("--")]
    n = int(args[0]) if args else 6
    mod = load_agent()
    if small_search:
        # tiny budgets: search on every eligible move, fast wall-clock
        mod._configure_search(cap_s=0.6, k_trees=2, sims_per_tree=8,
                              bank_s=10_000.0)
    # The agent's OWN deck is what it declares at deck selection; the engine
    # deals it that deck, so the smoke must play my_deck = the declared deck.
    my_deck = mod.agent({"select": None, "current": None, "logs": []})
    assert len(my_deck) == 60
    opp_deck = load_sample_deck()   # a fixed opponent deck for the smoke
    wins, done, all_lat = 0, 0, []
    for g in range(n):
        my_seat = g % 2
        mod.agent({"select": None, "current": None, "logs": []})
        if hasattr(mod, "_reseed"):
            mod._reseed(1000 + g)
        result, lat = play(mod, my_seat, my_deck, opp_deck, seed=1000 + g)
        done += 1
        if result == my_seat:
            wins += 1
        all_lat += lat
    all_lat.sort()
    p50 = all_lat[len(all_lat) // 2]
    p95 = all_lat[int(len(all_lat) * 0.95)]
    print(f"games={done} wins={wins} winrate={wins/done:.3f} "
          f"moves={len(all_lat)} latency p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms")
    if small_search:
        t = mod._TELEM
        print(f"telemetry: searched={t['searched']} sims={t['sims']} "
              f"fallbacks={t['fallbacks']} search_time={t['search_time']:.1f}s")
        assert t["searched"] > 0, "search never ran in --small-search mode"
        assert p95 < 5.0, f"p95 latency {p95:.1f}s exceeds small-search bound"
    print("OK — all agent picks legal, all games completed" if done == n
          else "INCOMPLETE")
```

(Leave `load_agent` and `play` untouched.)

- [ ] **Step 2: Run both modes**

Run: `PYTHONIOENCODING=utf-8 python scripts/test_submission.py 4 --small-search`
Expected: `games=4 ... winrate=1.000` (vs random), `telemetry: searched=<positive> ...`, `OK — all agent picks legal, all games completed`. Should finish in a few minutes.

Run: `PYTHONIOENCODING=utf-8 python scripts/test_submission.py 2`
Expected: unchanged default behavior, `OK` line (search at default budgets — slower per move; 2 games only).

- [ ] **Step 3: Commit**

```bash
git add scripts/test_submission.py
git commit -m "smoke: forced small-budget search mode with telemetry assertions"
```

---

### Task 9: strength eval (`scripts/eval_search.py`) + ship gate

**Files:**
- Create: `scripts/eval_search.py`
- Modify: `.superpowers/sdd/progress.md` (append the eval verdict)

**Interfaces:**
- Consumes: two independently-loaded `submission_src/main.py` module instances (Task 6 knobs `_configure_search`, `_reseed`, `_SEARCH_ENABLED`, `_DECK`, `_TELEM`); `ptcg.engine.BattleSession`.
- Produces: the ship-gate verdict — search agent vs raw-policy agent win rate with a Wilson 95% CI; PASS requires `wr >= 0.55` **and** `ci_lo > 0.50` (user-agreed gate).

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_search.py`:

```python
"""Ship-gate eval: search agent vs raw-policy agent, mirror decks.

Both sides are independent loads of submission_src/main.py (module state
isolated); side A searches, side B has _SEARCH_ENABLED=False. Seats
alternate per game. Workers run whole games sequentially (one battle per
process at a time); the parent aggregates and prints a Wilson 95% CI.

Gate (user-agreed): wr >= 0.55 AND ci_lo > 0.50.

Usage:
  python scripts/eval_search.py --games 400 --workers 3 --cap 0.8 --k 3 --sims 24
  python scripts/eval_search.py --games 20 --workers 1 --full          # full budgets
  python scripts/eval_search.py --games 60 --workers 3 --opp-deck dragapult-ex
"""
import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_main(name):
    path = os.path.join(REPO, "submission_src", "main.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _play_chunk(args):
    (games, seed0, cap, k, sims, full, opp_deck_slug) = args
    from ptcg.engine import BattleSession

    a = _load_main("submission_main_search")     # searcher
    b = _load_main("submission_main_policy")     # raw policy
    b._SEARCH_ENABLED = False
    if not full:
        a._configure_search(cap_s=cap, k_trees=k, sims_per_tree=sims,
                            bank_s=10_000.0)
    deck_a = a.agent({"select": None, "current": None, "logs": []})
    if opp_deck_slug:
        p = os.path.join(REPO, "decks", opp_deck_slug, "deck.csv")
        with open(p) as f:
            b._DECK = [int(r) for r in f.read().split("\n") if r.strip()][:60]
    deck_b = b.agent({"select": None, "current": None, "logs": []})
    wins = 0
    for g in range(games):
        seat_a = g % 2
        a.agent({"select": None, "current": None, "logs": []})
        b.agent({"select": None, "current": None, "logs": []})
        a._reseed(seed0 + 2 * g)
        random.seed(seed0 + 2 * g + 1)
        s = BattleSession(deck_a if seat_a == 0 else deck_b,
                          deck_b if seat_a == 0 else deck_a)
        try:
            while not s.done:
                mod = a if s.select_player == seat_a else b
                picks = mod.agent(s.obs)
                assert mod._is_legal(picks, s.obs["select"]), picks
                s.select(picks)
            if s.result == seat_a:
                wins += 1
        finally:
            s.close()
    return {"wins": wins, "games": games,
            "searched": a._TELEM["searched"], "sims": a._TELEM["sims"],
            "fallbacks": a._TELEM["fallbacks"],
            "search_time": a._TELEM["search_time"]}


def _wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cap", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--full", action="store_true",
                    help="default budgets (no reduced-cap override)")
    ap.add_argument("--opp-deck", default="",
                    help="portfolio deck slug for the raw-policy opponent")
    args = ap.parse_args()

    per = args.games // args.workers
    rem = args.games - per * args.workers
    jobs = []
    for w in range(args.workers):
        n = per + (1 if w < rem else 0)
        if n:
            jobs.append((n, 50_000 + 10_000 * w, args.cap, args.k, args.sims,
                         args.full, args.opp_deck))
    if len(jobs) == 1:
        stats = [_play_chunk(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(_play_chunk, jobs)
    wins = sum(s["wins"] for s in stats)
    games = sum(s["games"] for s in stats)
    lo, hi = _wilson(wins, games)
    wr = wins / max(games, 1)
    searched = sum(s["searched"] for s in stats)
    sims = sum(s["sims"] for s in stats)
    st = sum(s["search_time"] for s in stats)
    fb = sum(s["fallbacks"] for s in stats)
    print(f"search-vs-policy: games={games} wins={wins} wr={wr:.3f} "
          f"wilson95=[{lo:.3f},{hi:.3f}]")
    print(f"search telemetry: searched={searched} sims={sims} "
          f"fallbacks={fb} search_time={st:.0f}s "
          f"({st/max(searched,1):.2f}s/searched-move)")
    gate = wr >= 0.55 and lo > 0.50
    print("GATE:", "PASS" if gate else "FAIL",
          "(need wr>=0.55 and ci_lo>0.50)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick sanity run (small)**

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_search.py --games 12 --workers 3 --cap 0.5 --k 2 --sims 12`
Expected: completes in ~10–20 min, prints `search-vs-policy: games=12 ...` + telemetry + a GATE line (verdict at n=12 is noise — this step only proves the harness works, both agents legal, no crashes, `fallbacks` not dominating).

- [ ] **Step 3: Commit the harness**

```bash
git add scripts/eval_search.py
git commit -m "add search-vs-policy eval harness with wilson ci gate"
```

- [ ] **Step 4: Run the ship-gate eval (long, ~1.5–3 h)**

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_search.py --games 400 --workers 3 --cap 0.8 --k 3 --sims 24`
Expected: `GATE: PASS` with `wr >= 0.55` and `ci_lo > 0.50`. (At n=400 the CI half-width is ~0.049, so wr must be ≥ ~0.55 for the CI to clear 0.50 — the gate self-enforces.)

- [ ] **Step 5: Full-budget + non-mirror sanity samples**

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_search.py --games 20 --workers 1 --full`
Expected: no crashes/illegal picks; wr directionally consistent (small n — record, don't gate); search_time/searched-move well under the 20 s cap.

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_search.py --games 60 --workers 3 --opp-deck dragapult-ex`
Expected: completes clean; record wr (search agent on the sample deck vs raw policy on dragapult — generality signal, not a gate).

- [ ] **Step 6: Rebuild the shipping bundles and re-smoke**

Run: `PYTHONIOENCODING=utf-8 python scripts/make_submission.py && PYTHONIOENCODING=utf-8 python scripts/make_submission.py --no-search`
Expected: both `self-contained OK` lines again (search build also prints `search arena OK`).

Run: `PYTHONIOENCODING=utf-8 python scripts/test_submission.py 2 --small-search`
Expected: `OK — all agent picks legal, all games completed`, `searched > 0`.

- [ ] **Step 7: Record the verdict and commit**

Append to `.superpowers/sdd/progress.md` under a new `=== PHASE 4 (plan 2026-07-09-inference-search-phase4.md) ===` header: per-task status lines (written during execution) plus the final eval numbers (wr, CI, telemetry, full-budget + non-mirror results) and the ship/no-ship decision.

```bash
git add .superpowers/sdd/progress.md
git commit -m "phase-4 eval verdict and ledger update"
```

If the gate FAILS: do not ship; record the numbers, then investigate in order — (1) fallback rate (search erroring → policy would show `fallbacks` high), (2) determinization acceptance (K shrinking), (3) budget starvation (sims/searched-move < ~10), (4) c_puct/K/sims retune at the same wall cost. Re-run Step 4 after any fix.

---

## Plan Self-Review (performed while writing)

- **Spec coverage:** simsearch wrapper (Task 2 = spec §simsearch), determinizer layers/fallbacks (Task 3 = §Determinizer), PUCT/negamax/vote/chance handling (Tasks 4–5 = §Search), clock/trivial/floor/process-lifetime bank (Task 1 + Task 6 = §Clock), agent guards/telemetry/reset (Task 6 = §Agent integration), bundle + `--no-search` + AgentStart check (Task 7 = §Packaging), forced-search smoke (Task 8) and ship gate + full-budget + non-mirror samples (Task 9 = §Testing/acceptance). Out-of-scope items (batched leaf eval, manual_coin, single-tree IS-MCTS, temperature) are not planned — matching the spec.
- **Type consistency:** `SearchSession.begin/step -> tuple[int, dict] | None` used identically in Tasks 2/4/5/6; `Determinization` field names match between Tasks 3 and 4 (`opp_decklist` consumed by `search_move` roots); `search_move(..., tslice)` signature identical in Tasks 4/5/6; `_TELEM` keys match Tasks 6/8/9; `_configure_search` kwargs match Tasks 6/8/9.
- **Placeholder scan:** every code step carries complete, committable code; no TBD/TODO/"similar to Task N" anywhere.
