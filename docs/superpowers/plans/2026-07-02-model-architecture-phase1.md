# PTCG model architecture (phase 1) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the network and its data interface from
`docs/superpowers/specs/2026-07-02-model-architecture-design.md`, proven by an
untrained model playing fully legal games against the real engine and by a
PPO-shaped gradient step whose replayed log-probs match the actor's exactly.

**Architecture:** Entity-transformer encoder (pure state encoder) over tokens built
by an env-wrapper featurizer + belief tracker; shallow cross-attention decoder with
a flat pointer head over the engine's enumerated options; privileged critic as a
separate full-information encoder; composite within-select actions with a shared
sample/replay pick loop.

**Tech Stack:** Python ≥3.11, PyTorch ≥2.4 (CPU is sufficient for all tests),
NumPy, pytest. Game engine: the competition's `cg` ctypes package (already in the
repo).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-02-model-architecture-design.md`. On any
  conflict, the spec wins; deviations must be listed in the final report.
- Never modify anything under `pokemon-tcg-ai-battle/`. Never use the repo-root
  `deck.csv` (incomplete). The known-good 60-card deck is
  `pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv`.
- Engine package dir (importable `cg`):
  `pokemon-tcg-ai-battle/sample_submission/sample_submission`, overridable via env
  var `PTCG_ENGINE_DIR`.
- One battle per process (engine stores `battle_ptr` as a class attribute).
- Engine RNG is not seedable: engine-in-the-loop tests assert invariants and
  ranges, never exact game outcomes.
- The featurizer is a pure function of (observation dict, tracker snapshot, own
  deck list). No hidden state outside `BeliefTracker`.
- Trunk never receives select-context input (spec: pure state encoder).
- No masking of illegal actions anywhere except the pick loop's
  already-picked/done rules — the enumerated option list is the only legality
  oracle.
- All enum-id embeddings get hash-bucket fallback rows (`HASH_ROWS = 8`):
  `row = v if v < known else known + v % HASH_ROWS`.
- Deviations from spec accepted for phase 1: text-encoder embedding init
  (spec-optional) and the archetype-cluster aux head (needs training-time
  pseudo-labels) are deferred to the training-pipeline plan. Privileged deck/prize
  partitions are contingent on the Task 6 spike; fallback is both-hands privileged
  info, which must be recorded in the spec if hit.
- Commit style: short lowercase messages, commit directly on `main` (repo
  convention).

## File Structure

```
pyproject.toml            — package + deps + pytest config
ptcg/__init__.py          — empty
ptcg/engine.py            — engine locator, lean obs dicts, BattleSession, random policy
ptcg/cards.py             — static card/attack tables, id→row mapping, reserved rows
ptcg/tracker.py           — BeliefTracker (per-seat, membership multisets from logs)
ptcg/featurize.py         — constants, TokenizedState, EncodedSelect, public + privileged featurizers
ptcg/model.py             — ModelConfig, encoder trunk, decoder/pointer head, value+aux heads, critic
ptcg/action.py            — shared pick loop: sample_select / replay_select
ptcg/rollout.py           — Step/Trajectory containers, play_game()
tests/conftest.py         — engine fixtures, sample deck, tiny-model fixtures
tests/test_engine.py      … tests/test_ppo_smoke.py (one file per task, named below)
benchmarks/bench_featurizer.py — featurizer µs/select microbenchmark
```

---

### Task 1: Package scaffold + engine session wrapper

**Files:**
- Create: `pyproject.toml`, `ptcg/__init__.py`, `ptcg/engine.py`,
  `tests/conftest.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: repo engine at `pokemon-tcg-ai-battle/sample_submission/sample_submission`.
- Produces: `ptcg.engine.BattleSession(deck0: list[int], deck1: list[int])` with
  `.obs: dict`, `.select_player: int`, `.result: int` (−1 ongoing, 0/1 winner, 2
  draw), `.done: bool`, `.select(picks: list[int]) -> dict`, `.close()`;
  `ptcg.engine.random_picks(obs: dict, rng: random.Random) -> list[int]`;
  `ptcg.engine.load_sample_deck() -> list[int]`; `ptcg.engine.engine_dir() -> str`.
  All observations are plain dicts (tier-b parsing — never
  `to_observation_class`).

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "ptcg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.4"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["ptcg"]
```

- [ ] **Step 2: Write the failing test**

`tests/conftest.py`:

```python
import random
import pytest


@pytest.fixture(scope="session")
def sample_deck():
    from ptcg.engine import load_sample_deck
    return load_sample_deck()


@pytest.fixture()
def rng():
    return random.Random(7)
```

`tests/test_engine.py`:

```python
import pytest
from ptcg.engine import BattleSession, random_picks, load_sample_deck


def test_sample_deck_loads():
    deck = load_sample_deck()
    assert len(deck) == 60
    assert all(isinstance(c, int) and c > 0 for c in deck)


def test_full_random_game(sample_deck, rng):
    s = BattleSession(sample_deck, list(sample_deck))
    try:
        n = 0
        while not s.done:
            assert n < 5000, "selection cap hit"
            s.select(random_picks(s.obs, rng))
            n += 1
        assert s.result in (0, 1, 2)
        assert 5 <= n <= 1000  # benchmark observed 13..219
    finally:
        s.close()


def test_one_session_per_process(sample_deck, rng):
    s = BattleSession(sample_deck, list(sample_deck))
    try:
        with pytest.raises(RuntimeError):
            BattleSession(sample_deck, list(sample_deck))
    finally:
        s.close()
    s2 = BattleSession(sample_deck, list(sample_deck))  # reopen after close works
    s2.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/jame/Desktop/Pokemon-Kaggle && python3 -m pip install -e ".[dev]" && python3 -m pytest tests/test_engine.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'ptcg.engine'`

- [ ] **Step 4: Implement `ptcg/engine.py`**

```python
import os
import random
import sys

_game = None


def engine_dir() -> str:
    p = os.environ.get("PTCG_ENGINE_DIR")
    if p:
        return p
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo, "pokemon-tcg-ai-battle", "sample_submission", "sample_submission"
    )


def _load_game():
    global _game
    if _game is None:
        d = engine_dir()
        if not os.path.isdir(d):
            raise FileNotFoundError(f"engine dir not found: {d} (set PTCG_ENGINE_DIR)")
        if d not in sys.path:
            sys.path.insert(0, d)
        from cg import game  # loads the native library on import
        _game = game
    return _game


def load_sample_deck() -> list[int]:
    path = os.path.join(engine_dir(), "deck.csv")
    with open(path) as f:
        return [int(line) for line in f.read().split("\n")[:60]]


def random_picks(obs: dict, rng: random.Random) -> list[int]:
    sel = obs["select"]
    return rng.sample(range(len(sel["option"])), sel["maxCount"])


class BattleSession:
    """One battle per process: the engine keeps battle_ptr as global state."""

    _open = False

    def __init__(self, deck0: list[int], deck1: list[int]):
        if BattleSession._open:
            raise RuntimeError("a BattleSession is already open in this process")
        g = _load_game()
        obs, start = g.battle_start(list(deck0), list(deck1))
        if obs is None:
            raise ValueError(
                f"deck rejected: player={start.errorPlayer} type={start.errorType}"
            )
        BattleSession._open = True
        self._g = g
        self.obs = obs

    @property
    def select_player(self) -> int:
        return self.obs["current"]["yourIndex"]

    @property
    def result(self) -> int:
        return self.obs["current"]["result"]

    @property
    def done(self) -> bool:
        return self.result != -1

    def select(self, picks: list[int]) -> dict:
        self.obs = self._g.battle_select(list(picks))
        return self.obs

    def close(self) -> None:
        if BattleSession._open:
            self._g.battle_finish()
            BattleSession._open = False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engine.py -v`
Expected: 3 PASS (first run takes a few seconds: native lib load + full game)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml ptcg/__init__.py ptcg/engine.py tests/conftest.py tests/test_engine.py
git commit -m "ptcg package scaffold and engine session wrapper"
```

---

### Task 2: Static card and attack tables

**Files:**
- Create: `ptcg/cards.py`
- Test: `tests/test_cards.py`

**Interfaces:**
- Consumes: `ptcg.engine.engine_dir()` (to import `cg.api`).
- Produces: `ptcg.cards.CardTables` with `.n_rows: int`, `.attr: np.ndarray
  [n_rows, ATTR_DIM] float32`, `.n_attack_rows: int`, `.attack_feat: np.ndarray
  [n_attack_rows, ATK_DIM] float32`; `ptcg.cards.card_row(card_id, n_rows) -> int`;
  `ptcg.cards.attack_row(attack_id, tables) -> int`;
  `ptcg.cards.build_tables() -> CardTables` (cached);
  constants `PAD_ROW = 0`, `UNK_ROW = 1`, `N_RESERVED = 2`, `HEADROOM = 256`,
  `ATTR_DIM = 28`, `ATK_DIM = 14`.

- [ ] **Step 1: Write the failing test**

`tests/test_cards.py`:

```python
import numpy as np
from ptcg.cards import (
    ATTR_DIM, PAD_ROW, UNK_ROW, build_tables, card_row,
    A_HP, A_BASIC, A_STAGE1, A_STAGE2, A_EX, A_IS_BASIC_ENERGY,
)


def test_tables_shape_and_reserved_rows():
    t = build_tables()
    assert t.attr.shape == (t.n_rows, ATTR_DIM)
    assert t.attr.dtype == np.float32
    assert np.all(t.attr[PAD_ROW] == 0) and np.all(t.attr[UNK_ROW] == 0)


def test_gardevoir_line_stages():
    t = build_tables()
    r745, r746, r747 = (t.attr[card_row(c, t.n_rows)] for c in (745, 746, 747))
    assert r745[A_BASIC] == 1 and r745[A_STAGE1] == 0
    assert r746[A_STAGE1] == 1
    assert r747[A_STAGE2] == 1 and r747[A_EX] == 1 and r747[A_HP] > 0


def test_basic_energy_flag_and_unknown_id():
    t = build_tables()
    assert t.attr[card_row(3, t.n_rows)][A_IS_BASIC_ENERGY] == 1  # Basic Water
    assert card_row(999999, t.n_rows) == UNK_ROW
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.cards'`

- [ ] **Step 3: Implement `ptcg/cards.py`**

```python
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .engine import _load_game

PAD_ROW, UNK_ROW, N_RESERVED = 0, 1, 2
HEADROOM = 256
N_ENERGY_TYPES = 12  # EnergyType 0..11

# attr slots
A_HP, A_RETREAT, A_BASIC, A_STAGE1, A_STAGE2, A_EX, A_MEGA, A_TERA, A_ACESPEC = range(9)
A_IS_POKEMON, A_IS_ITEM, A_IS_TOOL, A_IS_SUPPORTER, A_IS_STADIUM = range(9, 14)
A_IS_BASIC_ENERGY, A_IS_SPECIAL_ENERGY = 14, 15
A_ENERGY_TYPE0 = 16  # 16..27 one-hot by EnergyType
ATTR_DIM = A_ENERGY_TYPE0 + N_ENERGY_TYPES  # 28

K_DAMAGE, K_COST_TOTAL = 0, 1
K_COST_TYPE0 = 2  # 2..13 cost count by EnergyType
ATK_DIM = K_COST_TYPE0 + N_ENERGY_TYPES  # 14


@dataclass(frozen=True)
class CardTables:
    n_rows: int
    attr: np.ndarray
    n_attack_rows: int
    attack_feat: np.ndarray
    _attack_index: dict


def card_row(card_id: int, n_rows: int) -> int:
    r = card_id + N_RESERVED
    return r if 0 <= card_id and r < n_rows else UNK_ROW


def attack_row(attack_id: int, tables: CardTables) -> int:
    return tables._attack_index.get(attack_id, UNK_ROW)


@lru_cache(maxsize=1)
def build_tables() -> CardTables:
    _load_game()  # ensures cg is importable
    from cg import api

    cards = api.all_card_data()
    attacks = api.all_attack()

    n_rows = N_RESERVED + max(c.cardId for c in cards) + HEADROOM
    attr = np.zeros((n_rows, ATTR_DIM), dtype=np.float32)
    for c in cards:
        v = attr[card_row(c.cardId, n_rows)]
        v[A_HP] = (c.hp or 0) / 300.0
        v[A_RETREAT] = (c.retreatCost or 0) / 5.0
        v[A_BASIC], v[A_STAGE1], v[A_STAGE2] = float(c.basic), float(c.stage1), float(c.stage2)
        v[A_EX], v[A_MEGA], v[A_TERA], v[A_ACESPEC] = (
            float(c.ex), float(c.megaEx), float(c.tera), float(c.aceSpec)
        )
        ct = int(c.cardType)
        v[A_IS_POKEMON] = float(ct == 0)
        v[A_IS_ITEM] = float(ct == 1)
        v[A_IS_TOOL] = float(ct == 2)
        v[A_IS_SUPPORTER] = float(ct == 3)
        v[A_IS_STADIUM] = float(ct == 4)
        v[A_IS_BASIC_ENERGY] = float(ct == 5)
        v[A_IS_SPECIAL_ENERGY] = float(ct == 6)
        et = int(c.energyType)
        if 0 <= et < N_ENERGY_TYPES:
            v[A_ENERGY_TYPE0 + et] = 1.0

    n_attack_rows = N_RESERVED + len(attacks)
    attack_feat = np.zeros((n_attack_rows, ATK_DIM), dtype=np.float32)
    attack_index = {}
    for i, a in enumerate(attacks):
        row = N_RESERVED + i
        attack_index[a.attackId] = row
        attack_feat[row, K_DAMAGE] = (a.damage or 0) / 300.0
        attack_feat[row, K_COST_TOTAL] = len(a.energies) / 5.0
        for e in a.energies:
            if 0 <= int(e) < N_ENERGY_TYPES:
                attack_feat[row, K_COST_TYPE0 + int(e)] += 1.0 / 5.0

    return CardTables(n_rows, attr, n_attack_rows, attack_feat, attack_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cards.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/cards.py tests/test_cards.py
git commit -m "static card and attack feature tables"
```

---

### Task 3: Belief tracker

**Files:**
- Create: `ptcg/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: raw log dicts (`obs["logs"]`).
- Produces: `ptcg.tracker.BeliefTracker(my_index: int)` with
  `.update(logs: list[dict]) -> None` and `.snapshot() ->
  BeliefSnapshot(opp_hand: dict[int, int], opp_deck: dict[int, int],
  opp_hidden_pool: dict[int, int])` — multisets of card ids the opponent is known
  to hold in hand, deck, or "somewhere in hand∪deck" respectively.
- Log/area constants: `LOG_MOVE_CARD = 6`, `LOG_MOVE_CARD_REVERSE = 7`,
  `AREA_DECK = 1`, `AREA_HAND = 2` (from `cg/api.py` enums; unknown log types are
  ignored).

Semantics (membership-only, sound-by-construction):
- Visible `MOVE_CARD` involving the opponent: `toArea == HAND` increments
  `opp_hand[cardId]`; `fromArea == HAND` decrements it if present (else it came
  from the unrevealed part). Same rule pair for `DECK`/`opp_deck`.
- Facedown `MOVE_CARD_REVERSE` out of opponent `HAND` or `DECK` into the other
  hidden zone: merge that zone's entire revealed multiset into `opp_hidden_pool`
  (we know membership of hand∪deck, not which). Facedown move into a visible
  zone: do nothing (the card re-reveals via state).
- Unknown log types or missing fields: ignore. Never raise.

- [ ] **Step 1: Write the failing test**

`tests/test_tracker.py`:

```python
from ptcg.tracker import BeliefTracker, LOG_MOVE_CARD, LOG_MOVE_CARD_REVERSE, AREA_DECK, AREA_HAND

AREA_DISCARD = 3


def mv(p, cid, fr, to):
    return {"type": LOG_MOVE_CARD, "playerIndex": p, "cardId": cid,
            "fromArea": fr, "toArea": to}


def mvr(p, fr, to):
    return {"type": LOG_MOVE_CARD_REVERSE, "playerIndex": p,
            "fromArea": fr, "toArea": to}


def test_tutor_reveal_then_play():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 112, AREA_DECK, AREA_HAND)])       # opp tutors Munkidori to hand
    assert t.snapshot().opp_hand == {112: 1}
    t.update([mv(1, 112, AREA_HAND, AREA_DISCARD)])    # opp plays/discards it
    assert t.snapshot().opp_hand == {}


def test_own_moves_ignored():
    t = BeliefTracker(my_index=0)
    t.update([mv(0, 112, AREA_DECK, AREA_HAND)])
    assert t.snapshot().opp_hand == {}


def test_facedown_hand_to_deck_demotes_to_pool():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 140, AREA_DECK, AREA_HAND)])
    t.update([mvr(1, AREA_HAND, AREA_DECK)])           # Iono-style facedown return
    s = t.snapshot()
    assert s.opp_hand == {} and s.opp_hidden_pool == {140: 1}


def test_unknown_log_type_ignored():
    t = BeliefTracker(my_index=0)
    t.update([{"type": 9999, "playerIndex": 1}, {"type": LOG_MOVE_CARD}])
    assert t.snapshot().opp_hand == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.tracker'`

- [ ] **Step 3: Implement `ptcg/tracker.py`**

```python
from collections import Counter
from dataclasses import dataclass

LOG_MOVE_CARD = 6
LOG_MOVE_CARD_REVERSE = 7
AREA_DECK = 1
AREA_HAND = 2
_HIDDEN = (AREA_DECK, AREA_HAND)


@dataclass(frozen=True)
class BeliefSnapshot:
    opp_hand: dict
    opp_deck: dict
    opp_hidden_pool: dict


class BeliefTracker:
    """Membership-only knowledge of the opponent's hidden zones, from one seat's logs."""

    def __init__(self, my_index: int):
        self.me = my_index
        self._hand = Counter()
        self._deck = Counter()
        self._pool = Counter()

    def _zone(self, area):
        return self._hand if area == AREA_HAND else self._deck

    def update(self, logs: list) -> None:
        for lg in logs or []:
            if lg.get("playerIndex") != 1 - self.me:
                continue
            t = lg.get("type")
            fr, to = lg.get("fromArea"), lg.get("toArea")
            if t == LOG_MOVE_CARD:
                cid = lg.get("cardId")
                if cid is None:
                    continue
                if fr in _HIDDEN:
                    z = self._zone(fr)
                    if z[cid] > 0:
                        z[cid] -= 1
                    elif self._pool[cid] > 0:
                        self._pool[cid] -= 1
                if to in _HIDDEN:
                    self._zone(to)[cid] += 1
            elif t == LOG_MOVE_CARD_REVERSE:
                if fr in _HIDDEN and to in _HIDDEN:
                    self._pool.update(self._zone(fr))
                    self._zone(fr).clear()

    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            {k: v for k, v in self._hand.items() if v > 0},
            {k: v for k, v in self._deck.items() if v > 0},
            {k: v for k, v in self._pool.items() if v > 0},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tracker.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/tracker.py tests/test_tracker.py
git commit -m "belief tracker with membership multisets"
```

---

### Task 4: Featurizer — token assembly

**Files:**
- Create: `ptcg/featurize.py`
- Test: `tests/test_featurize.py`

**Interfaces:**
- Consumes: `CardTables`/`card_row` (Task 2), `BeliefSnapshot` (Task 3), obs dicts
  (Task 1).
- Produces (consumed by Tasks 5–13):

```python
# ptcg/featurize.py constants
MAX_TOKENS = 192
NUM_DIM = 40
KIND_SPECIAL, KIND_ENTITY, KIND_CHILD, KIND_MULTISET = 0, 1, 2, 3
OWNER_SELF, OWNER_OPP, OWNER_NEUTRAL = 0, 1, 2
# zones: AreaType values 1..12 reused; 13=global, 14=player-summary, 15=value, 16=scratch
N_ZONE, N_OWNER, N_KIND, N_POS = 17, 3, 4, 16

@dataclass
class TokenizedState:
    card: np.ndarray      # [MAX_TOKENS] int64  (card table rows; PAD_ROW when none)
    numeric: np.ndarray   # [MAX_TOKENS, NUM_DIM] float32
    owner: np.ndarray     # [MAX_TOKENS] int64
    zone: np.ndarray      # [MAX_TOKENS] int64
    kind: np.ndarray      # [MAX_TOKENS] int64
    pos: np.ndarray       # [MAX_TOKENS] int64
    mask: np.ndarray      # [MAX_TOKENS] bool  (True = real token)
    ref: dict             # (playerIndex, area, index, sub) -> token row; sub=-1 main card
    mrow: dict            # (owner, zone, card_id) -> multiset token row
    n: int                # token count

def featurize_state(obs: dict, me: int, own_deck: list[int],
                    belief: BeliefSnapshot, tables: CardTables) -> TokenizedState
```

- Special token layout (fixed rows): 0 = global, 1–2 = player summaries
  (self, opp), 3–4 = value tokens, 5–8 = scratch. Entity/child/multiset tokens
  follow from row 9.
- Numeric slots (named constants in the module): `F_HP=0, F_MAXHP=1, F_DMG=2,
  F_COUNT=3, F_POISON=4, F_BURN=5, F_ASLEEP=6, F_PARA=7, F_CONF=8,
  F_APPEAR=9, F_ACTIVE=10, F_BENCHIX=11, F_DECKN=12, F_HANDN=13, F_PRIZEN=14,
  F_TURN=15, F_SUPPORTER=16, F_STADIUMF=17, F_ENERGYATT=18, F_RETREATED=19,
  F_SPLIT=20, F_ENERGY0..F_ENERGY11=21..32` (energy counts /5). Slots 33–39
  reserved zeros.
- Own-side zones: hand → one entity token per card; deck∪prizes union = own deck
  multiset minus every own visible card (hand, board incl. children and
  pre-evolutions, discard, own stadium, revealed own prizes), emitted as multiset
  tokens with `F_COUNT = count/4`; discard → multiset tokens.
- Opponent side: board/discard/stadium from state; `belief.opp_hand`,
  `belief.opp_deck`, `belief.opp_hidden_pool` as multiset tokens (zones HAND,
  DECK, and zone 13-global? no — pool uses zone `AREA_DECK` with
  `owner=OWNER_OPP` and `F_SPLIT=1.0` marking pool membership).
- Children: one token per attached energy card, tool, and pre-evolution card,
  `kind=KIND_CHILD`, `pos = child slot index (clamped to N_POS-1)`, `ref` keyed
  with sub = 100+energyIndex, 200+toolIndex, 300+preEvo index (encoding scheme
  constant `SUB_ENERGY=100, SUB_TOOL=200, SUB_PREEVO=300`).
- Truncation: if tokens would exceed `MAX_TOKENS`, drop multiset tokens with the
  smallest counts first and set `F_SPLIT` on the global token to the dropped
  fraction; never drop entity/child/special tokens (assert they fit).

- [ ] **Step 1: Write the failing test** — `tests/test_featurize.py` with three
  tests: (a) a hand-built minimal obs dict (one active Pokémon with one attached
  energy each side, two cards in own hand, empty discards) produces exactly the
  expected token count `9 + 2 + 2 + 2 + (60-4 distinct union rows)` and correct
  `ref` entries for active `(me, 4, 0, -1)` and its energy child `(me, 4, 0, 100)`;
  (b) every own visible card decrements the union multiset (play a hand-built obs
  where 4 copies of card X: 1 in hand, 1 attached — union row for X has
  `F_COUNT == 2/4`); (c) real-engine sweep: featurize every select in 3 full
  random games (reusing `BattleSession` + `random_picks`), asserting no exception,
  `mask.sum() == n`, all indices < MAX_TOKENS, and every option with a CARD-type
  area/index resolvable through `ref` or `mrow` (resolution helper lands in
  Task 5 — here assert board cards only: every in-play Pokémon of both players has
  a `ref` entry).

Test code:

```python
import numpy as np
import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import (
    MAX_TOKENS, F_COUNT, KIND_CHILD, SUB_ENERGY, featurize_state,
)
from ptcg.tracker import BeliefTracker


def _mini_obs(me=0):
    def pkm(cid, serial, energy_cid):
        return {"id": cid, "serial": serial, "hp": 100, "maxHp": 100,
                "appearThisTurn": False, "energies": [1],
                "energyCards": [{"id": energy_cid, "serial": serial + 500,
                                 "playerIndex": 0}],
                "tools": [], "preEvolution": []}
    def player(hand):
        return {"active": [pkm(745, 1, 1)], "bench": [], "benchMax": 5,
                "deckCount": 50, "discard": [], "prize": [None] * 6,
                "handCount": len(hand), "hand": hand, "poisoned": False,
                "burned": False, "asleep": False, "paralyzed": False,
                "confused": False}
    hand = [{"id": 746, "serial": 10, "playerIndex": 0},
            {"id": 747, "serial": 11, "playerIndex": 0}]
    p1 = player(hand)
    p2 = player(None)
    p2["hand"] = None
    return {"select": None, "logs": [],
            "current": {"turn": 3, "turnActionCount": 0, "yourIndex": me,
                        "firstPlayer": 0, "supporterPlayed": False,
                        "stadiumPlayed": False, "energyAttached": False,
                        "retreated": False, "result": -1, "stadium": [],
                        "looking": None, "players": [p1, p2]}}


def test_mini_obs_refs_and_children():
    tables = build_tables()
    deck = load_sample_deck()
    ts = featurize_state(_mini_obs(), 0, deck, BeliefTracker(0).snapshot(), tables)
    assert (0, 4, 0, -1) in ts.ref            # my active
    assert (0, 4, 0, SUB_ENERGY + 0) in ts.ref  # its attached energy
    child_row = ts.ref[(0, 4, 0, SUB_ENERGY + 0)]
    assert ts.kind[child_row] == KIND_CHILD
    assert ts.n == int(ts.mask.sum()) <= MAX_TOKENS


def test_union_multiset_decrements():
    tables = build_tables()
    deck = load_sample_deck()
    ts = featurize_state(_mini_obs(), 0, deck, BeliefTracker(0).snapshot(), tables)
    # deck.csv holds 4 copies of 722; none visible in mini obs -> count 4/4
    row = ts.mrow[(0, 1, 722)]
    assert np.isclose(ts.numeric[row, F_COUNT], 1.0)


def test_real_game_sweep():
    tables = build_tables()
    deck = load_sample_deck()
    rng = random.Random(3)
    for _ in range(3):
        s = BattleSession(deck, list(deck))
        trackers = [BeliefTracker(0), BeliefTracker(1)]
        try:
            while not s.done:
                me = s.select_player
                trackers[me].update(s.obs.get("logs", []))
                ts = featurize_state(s.obs, me, deck,
                                     trackers[me].snapshot(), tables)
                assert ts.n == int(ts.mask.sum()) <= MAX_TOKENS
                cur = s.obs["current"]
                for pi, pl in enumerate(cur["players"]):
                    for area, lst in ((4, pl["active"]), (5, pl["bench"])):
                        for i, pk in enumerate(lst):
                            if pk is not None:
                                assert (pi, area, i, -1) in ts.ref
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_featurize.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Implement token assembly in `ptcg/featurize.py`**

Implementation requirements (write the module to satisfy the interface block
above exactly):

```python
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .cards import PAD_ROW, CardTables, card_row
from .tracker import BeliefSnapshot

MAX_TOKENS = 192
NUM_DIM = 40
KIND_SPECIAL, KIND_ENTITY, KIND_CHILD, KIND_MULTISET = 0, 1, 2, 3
OWNER_SELF, OWNER_OPP, OWNER_NEUTRAL = 0, 1, 2
N_ZONE, N_OWNER, N_KIND, N_POS = 17, 3, 4, 16
Z_GLOBAL, Z_PSUM, Z_VALUE, Z_SCRATCH = 13, 14, 15, 16
SUB_ENERGY, SUB_TOOL, SUB_PREEVO = 100, 200, 300
(F_HP, F_MAXHP, F_DMG, F_COUNT, F_POISON, F_BURN, F_ASLEEP, F_PARA, F_CONF,
 F_APPEAR, F_ACTIVE, F_BENCHIX, F_DECKN, F_HANDN, F_PRIZEN, F_TURN,
 F_SUPPORTER, F_STADIUMF, F_ENERGYATT, F_RETREATED, F_SPLIT) = range(21)
F_ENERGY0 = 21  # ..32


@dataclass
class TokenizedState:
    card: np.ndarray
    numeric: np.ndarray
    owner: np.ndarray
    zone: np.ndarray
    kind: np.ndarray
    pos: np.ndarray
    mask: np.ndarray
    ref: dict = field(default_factory=dict)
    mrow: dict = field(default_factory=dict)
    n: int = 0


class _Builder:
    def __init__(self, tables: CardTables):
        self.t = tables
        self.s = TokenizedState(
            np.full(MAX_TOKENS, PAD_ROW, np.int64),
            np.zeros((MAX_TOKENS, NUM_DIM), np.float32),
            np.zeros(MAX_TOKENS, np.int64), np.zeros(MAX_TOKENS, np.int64),
            np.zeros(MAX_TOKENS, np.int64), np.zeros(MAX_TOKENS, np.int64),
            np.zeros(MAX_TOKENS, bool),
        )

    def add(self, card_id, owner, zone, kind, pos=0):
        s, i = self.s, self.s.n
        if i >= MAX_TOKENS:
            raise OverflowError("token budget exceeded")
        s.card[i] = card_row(card_id, self.t.n_rows) if card_id else PAD_ROW
        s.owner[i], s.zone[i], s.kind[i] = owner, zone, kind
        s.pos[i] = min(pos, N_POS - 1)
        s.mask[i] = True
        s.n += 1
        return i


def featurize_state(obs, me, own_deck, belief, tables) -> TokenizedState:
    b = _Builder(tables)
    cur = obs["current"]
    opp = 1 - me
    # specials: 0 global, 1-2 player summaries, 3-4 value, 5-8 scratch
    g = b.add(0, OWNER_NEUTRAL, Z_GLOBAL, KIND_SPECIAL)
    for pi in (me, opp):
        b.add(0, OWNER_SELF if pi == me else OWNER_OPP, Z_PSUM, KIND_SPECIAL)
    for _ in range(2):
        b.add(0, OWNER_NEUTRAL, Z_VALUE, KIND_SPECIAL)
    for k in range(4):
        b.add(0, OWNER_NEUTRAL, Z_SCRATCH, KIND_SPECIAL, pos=k)
    num = b.s.numeric
    num[g, F_TURN] = cur["turn"] / 50.0
    num[g, F_SUPPORTER] = float(cur["supporterPlayed"])
    num[g, F_STADIUMF] = float(cur["stadiumPlayed"])
    num[g, F_ENERGYATT] = float(cur["energyAttached"])
    num[g, F_RETREATED] = float(cur["retreated"])
    # ... player summaries: deck/hand/prize counts into F_DECKN/F_HANDN/F_PRIZEN
    # ... entities: per player active+bench via _add_pokemon (children inside)
    # ... stadium, own hand tokens, looking tokens
    # ... own visible Counter -> union = Counter(own_deck) - visible -> multisets
    # ... own+opp discard multisets, belief multisets (pool: F_SPLIT=1.0)
    return b.s
```

The elided `...` blocks are the mechanical application of the interface rules
above — implement them fully in this task (they are the deliverable, ~120 lines).
The trickiest helper in full, to remove ambiguity (AreaType values: ACTIVE=4,
ENERGY=8, TOOL=9, PRE_EVOLUTION=10):

```python
def _add_pokemon(b, pk, owner, player_index, area, idx):
    row = b.add(pk["id"], owner, area, KIND_ENTITY, pos=idx)
    s = b.s
    s.ref[(player_index, area, idx, -1)] = row
    n = s.numeric[row]
    mx = pk.get("maxHp") or 0
    n[F_HP] = (pk.get("hp") or 0) / 300.0
    n[F_MAXHP] = mx / 300.0
    n[F_DMG] = ((mx - (pk.get("hp") or 0)) / mx) if mx else 0.0
    n[F_APPEAR] = float(pk.get("appearThisTurn") or False)
    n[F_ACTIVE] = float(area == 4)
    n[F_BENCHIX] = idx / 8.0
    for e in pk.get("energies") or []:
        if 0 <= int(e) < 12:
            n[F_ENERGY0 + int(e)] += 0.2
    for sub, zone, key in ((SUB_ENERGY, 8, "energyCards"),
                           (SUB_TOOL, 9, "tools"),
                           (SUB_PREEVO, 10, "preEvolution")):
        for j, c in enumerate(pk.get(key) or []):
            r = b.add(c["id"], owner, zone, KIND_CHILD, pos=j)
            s.ref[(player_index, area, idx, sub + j)] = r
    return row
```

Special-condition booleans (`poisoned`…`confused`) live on `PlayerState` and
apply to the active Pokémon: set `F_POISON..F_CONF` on that player's active
token. `_own_union(own_deck, obs, me)` returns the deck∪prizes Counter after
subtracting every visible own card id (hand, board incl. children and
pre-evolutions, discard, own stadium, face-up own prizes); multiset emission
registers `mrow[(owner, zone, card_id)]` with `F_COUNT = count/4`. Truncation
rule: before emitting multisets, if `b.s.n + n_multiset > MAX_TOKENS`, drop
smallest-count rows first and write the dropped fraction to `num[g, F_SPLIT]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_featurize.py -v`
Expected: 3 PASS (sweep test runs 3 full games; a few seconds)

- [ ] **Step 5: Commit**

```bash
git add ptcg/featurize.py tests/test_featurize.py
git commit -m "featurizer token assembly"
```

---

### Task 5: Featurizer — option and query encoding

**Files:**
- Modify: `ptcg/featurize.py` (append)
- Test: `tests/test_options.py`

**Interfaces:**
- Consumes: `TokenizedState` (Task 4), obs `select` dicts, `CardTables`.
- Produces:

```python
HASH_ROWS = 8
N_SELECT_TYPE, N_SELECT_CTX, N_OPT_TYPE = 11 + HASH_ROWS, 49 + HASH_ROWS, 17 + HASH_ROWS
OPT_SCALAR_DIM, Q_SCALAR_DIM = 4, 6

@dataclass
class EncodedSelect:
    opt_type: np.ndarray    # [O] int64 (hash-bucketed OptionType)
    opt_ref: np.ndarray     # [O] int64 token row, or -1
    opt_card: np.ndarray    # [O] int64 card table row (0 when n/a)
    opt_attack: np.ndarray  # [O] int64 attack table row (0 when n/a)
    opt_scalar: np.ndarray  # [O, 4] float32: number/10, count/5, has_number, is_energy_unit
    q_type: int             # hash-bucketed SelectType
    q_ctx: int              # hash-bucketed SelectContext
    q_scalar: np.ndarray    # [6] float32: min/5, max/5, remainEnergyCost/5,
                            #   remainDamageCounter/10, has_deck_list, n_options/64
    q_ref: np.ndarray       # [2] int64: contextCard row, effect row (or -1)
    min_count: int
    max_count: int

def hash_id(v: int, known: int) -> int
def encode_select(obs: dict, ts: TokenizedState, tables: CardTables) -> EncodedSelect
```

- Reference resolution order for an option: (1) `(playerIndex, area, index, sub)`
  in `ts.ref` (sub from energyIndex/toolIndex when present); (2) the card list the
  index points into (`select.deck`, `looking`, hand, discard, prize of that
  player) → card id → `ts.mrow` or hand-token `ref`; (3) `-1` with
  `opt_card` set from the resolved card id when known, else `PAD_ROW`. `PLAY`
  options (`index` into own hand) resolve via `ref[(me, 2, index, -1)]`.
  `ATTACK` options set `opt_attack`. Unknown `OptionType` values hash-bucket via
  `hash_id`. Never raise on unseen enum members.

- [ ] **Step 1: Write the failing test** — `tests/test_options.py`:

```python
import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import (
    N_OPT_TYPE, encode_select, featurize_state, hash_id,
)
from ptcg.tracker import BeliefTracker


def test_hash_id_buckets():
    assert hash_id(5, 17) == 5
    assert 17 <= hash_id(23, 17) < 17 + 8
    assert hash_id(23, 17) == hash_id(23 + 8, 17)


def test_unknown_option_type_no_crash():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        ts = featurize_state(s.obs, s.select_player, deck,
                             BeliefTracker(s.select_player).snapshot(), tables)
        sel = {**s.obs["select"], "option": [{"type": 9999}]}
        es = encode_select({**s.obs, "select": sel}, ts, tables)
        assert es.opt_type[0] >= 17 and es.opt_type[0] < N_OPT_TYPE
    finally:
        s.close()


def test_real_game_option_sweep():
    tables = build_tables()
    deck = load_sample_deck()
    rng = random.Random(11)
    for _ in range(5):
        s = BattleSession(deck, list(deck))
        trackers = [BeliefTracker(0), BeliefTracker(1)]
        try:
            while not s.done:
                me = s.select_player
                trackers[me].update(s.obs.get("logs", []))
                ts = featurize_state(s.obs, me, deck,
                                     trackers[me].snapshot(), tables)
                es = encode_select(s.obs, ts, tables)
                o = len(s.obs["select"]["option"])
                assert es.opt_type.shape == (o,)
                assert es.min_count <= es.max_count <= o
                assert (es.opt_ref < ts.n).all()
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_options.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode_select'`

- [ ] **Step 3: Implement `encode_select` + `hash_id`** appended to
  `ptcg/featurize.py`, following the interface block exactly (~90 lines). Key
  code:

```python
def hash_id(v: int, known: int) -> int:
    v = int(v) if v is not None else 0
    return v if 0 <= v < known else known + (v % HASH_ROWS)


def _resolve(opt: dict, obs: dict, ts: TokenizedState, tables) -> tuple[int, int]:
    """-> (token_row or -1, card table row)"""
    pi, area, idx = opt.get("playerIndex"), opt.get("area"), opt.get("index")
    for subkey, base in (("energyIndex", SUB_ENERGY), ("toolIndex", SUB_TOOL)):
        if opt.get(subkey) is not None and (pi, area, idx, base + opt[subkey]) in ts.ref:
            return ts.ref[(pi, area, idx, base + opt[subkey])], PAD_ROW
    if (pi, area, idx, -1) in ts.ref:
        return ts.ref[(pi, area, idx, -1)], PAD_ROW
    cid = _card_id_at(opt, obs)          # walks select.deck / looking / zone lists
    if cid is not None:
        row = ts.mrow.get((_owner_of(pi, obs), area, cid), -1)
        return row, card_row(cid, tables.n_rows)
    return -1, PAD_ROW
```

`_card_id_at` handles: `select.deck` lists (when `SelectData.deck` is not None
and area is DECK/LOOKING), `looking`, both discards, revealed prizes, own hand —
indexing with `opt["index"]` guarded by bounds checks, returning `None` on any
miss. `PLAY` options (`type == 7`) resolve as `(me, AREA_HAND=2, opt["index"], -1)`.
`ATTACK` options set `opt_attack = attack_row(opt["attackId"], tables)`.
`opt_scalar` and `q_scalar` fill exactly the slots named in the interface block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_options.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/featurize.py tests/test_options.py
git commit -m "option and query encoding with hash-bucket fallbacks"
```

---

### Task 6: Privileged featurizer + hidden-info capability spike

**Files:**
- Modify: `ptcg/featurize.py` (append `featurize_privileged`)
- Test: `tests/test_privileged.py`
- Create: `benchmarks/spike_visualize_data.py` (throwaway probe, committed for the record)

**Interfaces:**
- Produces: `featurize_privileged(obs_a: dict, obs_b: dict, decks: tuple,
  tables: CardTables) -> TokenizedState` — full-information tokenization built
  from BOTH seats' most recent observations of the same state: each player's hand
  becomes entity tokens (owner-correct), both deck∪prize unions as multisets. No
  belief tracker input (nothing is hidden that we can supply).
- Spike output: a short markdown note appended to this plan's PR/commit message
  documenting whether `cg.game.visualize_data()` exposes true deck order or prize
  identities.

- [ ] **Step 1: Spike — probe VisualizeData**

`benchmarks/spike_visualize_data.py`:

```python
"""Probe: does VisualizeData expose deck order / prize identity? Run once, read output."""
import json
import random
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from cg.game import visualize_data

deck = load_sample_deck()
s = BattleSession(deck, list(deck))
rng = random.Random(0)
for _ in range(10):
    if s.done:
        break
    s.select(random_picks(s.obs, rng))
d = json.loads(visualize_data())
print(json.dumps(d, indent=1)[:4000])
s.close()
```

Run: `python3 benchmarks/spike_visualize_data.py`
Decision rule: if the JSON contains per-card identities for prizes or deck order,
extend `featurize_privileged` to consume it (add a `viz: dict | None` parameter);
if not, proceed with both-hands privileged info only and record the limitation in
the spec's privileged-critic section (edit
`docs/superpowers/specs/2026-07-02-model-architecture-design.md` accordingly —
one sentence).

- [ ] **Step 2: Write the failing test** — `tests/test_privileged.py`:

```python
import random
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import OWNER_OPP, featurize_privileged, featurize_state
from ptcg.tracker import BeliefTracker

AREA_HAND_Z = 2  # AreaType.HAND, used as the zone id for hand tokens


def test_privileged_sees_both_hands():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    rng = random.Random(5)
    try:
        last = [None, None]
        while not s.done and (last[0] is None or last[1] is None):
            last[s.select_player] = s.obs
            s.select(random_picks(s.obs, rng))
        assert last[0] is not None and last[1] is not None
        # privileged view is built from seat 0's perspective: OWNER_OPP == seat 1
        pv = featurize_privileged(last[0], last[1], (deck, list(deck)), tables)
        opp_hand_rows = [
            i for i in range(pv.n)
            if pv.zone[i] == AREA_HAND_Z and pv.owner[i] == OWNER_OPP
        ]
        expected = last[1]["current"]["players"][1]["handCount"]
        assert len(opp_hand_rows) == expected
        # the public view from seat 0 must contain no opponent hand tokens
        pub = featurize_state(last[0], 0, deck, BeliefTracker(0).snapshot(), tables)
        assert [
            i for i in range(pub.n)
            if pub.zone[i] == AREA_HAND_Z and pub.owner[i] == OWNER_OPP
        ] == []
    finally:
        s.close()
```

- [ ] **Step 3: Run test to verify it fails, implement, re-run**

Run: `python3 -m pytest tests/test_privileged.py -v` → FAIL (`ImportError`).
Implement `featurize_privileged` by reusing `_Builder` and `_add_pokemon`:
identical to `featurize_state` from seat-0's perspective except (a) seat-1's
`hand` list is taken from `obs_b` and emitted as opponent-owned hand entity
tokens, (b) both unions are computed from both decks, (c) no belief multisets.
Re-run → PASS.

- [ ] **Step 4: Commit**

```bash
git add ptcg/featurize.py tests/test_privileged.py benchmarks/spike_visualize_data.py
git commit -m "privileged featurizer and hidden-info spike"
```

---

### Task 7: Model — embeddings and trunk

**Files:**
- Create: `ptcg/model.py`
- Test: `tests/test_model.py`
- Modify: `tests/conftest.py` (add tiny-config fixture)

**Interfaces:**
- Consumes: `TokenizedState` arrays, `CardTables` dims, featurize constants.
- Produces:

```python
@dataclass
class ModelConfig:
    d: int = 512; layers: int = 8; heads: int = 8; ffn: int = 2048
    dec_layers: int = 2
    n_card_rows: int = 0        # filled from CardTables
    n_attack_rows: int = 0
    # vocab sizes imported from featurize: N_OWNER, N_ZONE, N_KIND, N_POS,
    # N_SELECT_TYPE, N_SELECT_CTX, N_OPT_TYPE; dims ATTR_DIM, ATK_DIM, NUM_DIM,
    # OPT_SCALAR_DIM, Q_SCALAR_DIM

def teacher_config(tables) -> ModelConfig      # d512 L8 H8 ffn2048 dec2
def student_config(tables) -> ModelConfig      # d224 L4 H8 ffn896 dec1
def tiny_config(tables) -> ModelConfig         # d64 L2 H4 ffn128 dec1 (tests)

class Encoder(nn.Module):
    def forward(self, batch: dict[str, Tensor]) -> Tensor  # [B, MAX_TOKENS, d]
    # batch keys: card, numeric, owner, zone, kind, pos (int64/float32), mask (bool)

def collate_states(states: list[TokenizedState]) -> dict[str, Tensor]
```

Value/scratch token rows are fixed (3–4 value, 5–8 scratch per Task 4 layout);
export `VALUE_ROWS = (3, 4)` from `ptcg/model.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_model.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.model import Encoder, collate_states, teacher_config, tiny_config


def _real_states(n=4):
    import random
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.featurize import featurize_state
    from ptcg.tracker import BeliefTracker
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    rng = random.Random(1)
    out = []
    try:
        while len(out) < n and not s.done:
            me = s.select_player
            out.append(featurize_state(s.obs, me, deck,
                                       BeliefTracker(me).snapshot(), tables))
            s.select(random_picks(s.obs, rng))
    finally:
        s.close()
    return tables, out


def test_forward_shapes_and_grads():
    tables, states = _real_states()
    cfg = tiny_config(tables)
    enc = Encoder(cfg)
    batch = collate_states(states)
    h = enc(batch)
    assert h.shape == (len(states), batch["card"].shape[1], cfg.d)
    h.sum().backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


def test_padding_invariance():
    tables, states = _real_states(2)
    enc = Encoder(tiny_config(tables)).eval()
    batch = collate_states(states)
    with torch.no_grad():
        h1 = enc(batch)
        batch2 = {k: v.clone() for k, v in batch.items()}
        batch2["card"][~batch2["mask"]] = 1  # scribble on padding
        h2 = enc(batch2)
    m = batch["mask"]
    assert torch.allclose(h1[m], h2[m], atol=1e-5)


def test_teacher_param_count():
    tables, _ = _real_states(1)
    n = sum(p.numel() for p in Encoder(teacher_config(tables)).parameters())
    assert 20_000_000 < n < 32_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.model'`

- [ ] **Step 3: Implement encoder in `ptcg/model.py`**

```python
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import featurize as F
from .cards import ATK_DIM, ATTR_DIM

VALUE_ROWS = (3, 4)


@dataclass
class ModelConfig:
    d: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 2048
    dec_layers: int = 2
    dropout: float = 0.0  # nonzero would break padding invariance and the ratio contract
    n_card_rows: int = 0
    n_attack_rows: int = 0


def teacher_config(tables):
    return ModelConfig(n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


def student_config(tables):
    return ModelConfig(d=224, layers=4, heads=8, ffn=896, dec_layers=1,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


def tiny_config(tables):
    return ModelConfig(d=64, layers=2, heads=4, ffn=128, dec_layers=1,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


class CardEmbed(nn.Module):
    """Shared card-identity encoding: id embedding + printed-attribute projection."""

    def __init__(self, cfg, attr_table: np.ndarray):
        super().__init__()
        self.emb = nn.Embedding(cfg.n_card_rows, cfg.d, padding_idx=0)
        self.register_buffer("attr", torch.from_numpy(attr_table))
        self.attr_proj = nn.Linear(ATTR_DIM, cfg.d)

    def forward(self, rows):
        return self.emb(rows) + self.attr_proj(self.attr[rows])


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None):
        super().__init__()
        if attr_table is None:
            from .cards import build_tables
            attr_table = build_tables().attr
        self.cfg = cfg
        self.card = CardEmbed(cfg, attr_table)
        self.num = nn.Linear(F.NUM_DIM, cfg.d)
        self.owner = nn.Embedding(F.N_OWNER, cfg.d)
        self.zone = nn.Embedding(F.N_ZONE, cfg.d)
        self.kind = nn.Embedding(F.N_KIND, cfg.d)
        self.pos = nn.Embedding(F.N_POS, cfg.d)
        self.norm = nn.LayerNorm(cfg.d)
        layer = nn.TransformerEncoderLayer(
            cfg.d, cfg.heads, cfg.ffn, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, cfg.layers)

    def forward(self, batch):
        x = (self.card(batch["card"]) + self.num(batch["numeric"])
             + self.owner(batch["owner"]) + self.zone(batch["zone"])
             + self.kind(batch["kind"]) + self.pos(batch["pos"]))
        x = self.norm(x)
        return self.trunk(x, src_key_padding_mask=~batch["mask"])


def collate_states(states):
    def stack(name, dtype):
        return torch.stack([torch.as_tensor(getattr(s, name)).to(dtype) for s in states])
    return {
        "card": stack("card", torch.int64),
        "numeric": stack("numeric", torch.float32),
        "owner": stack("owner", torch.int64),
        "zone": stack("zone", torch.int64),
        "kind": stack("kind", torch.int64),
        "pos": stack("pos", torch.int64),
        "mask": stack("mask", torch.bool),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: 3 PASS (teacher param-count test builds a 27M-param model on CPU; ~10 s)

- [ ] **Step 5: Commit**

```bash
git add ptcg/model.py tests/test_model.py
git commit -m "card embeddings and transformer trunk"
```

---

### Task 8: Model — decoder, pointer head, value and aux heads

**Files:**
- Modify: `ptcg/model.py` (append), `tests/test_model.py` (append)

**Interfaces:**
- Produces:

```python
def collate_selects(selects: list[EncodedSelect], device=None) -> dict[str, Tensor]
    # pads options to batch max O; adds "opt_mask" [B, O] bool

class PolicyModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None, attack_table=None)
    def encode(self, state_batch) -> Tensor                       # Encoder forward
    def option_logits(self, trunk, state_batch, sel_batch,
                      picked: Tensor) -> Tensor                    # [B, O+1]; col O = done
    def public_value(self, trunk) -> Tensor                        # [B] in (-1, 1)
    def prize_diff(self, trunk) -> Tensor                          # [B]
    def aux_decklist(self, trunk) -> Tensor                        # [B, n_card_rows] softplus rates
    def aux_hand(self, trunk) -> Tensor                            # [B, n_card_rows] softplus rates
    # picked: [B, O+1] bool of already-picked options (True -> masked to -inf);
    # done column masked to -inf where len(picked) < min_count;
    # padded option columns always -inf.
```

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model.py`):

```python
def test_option_logits_masking():
    import numpy as np
    from ptcg.featurize import encode_select, featurize_state
    from ptcg.model import PolicyModel, collate_selects, collate_states, tiny_config
    tables, states = _real_states(1)
    # rebuild the matching select for the same obs — reuse helper game
    import random
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.tracker import BeliefTracker
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        me = s.select_player
        ts = featurize_state(s.obs, me, deck, BeliefTracker(me).snapshot(), tables)
        es = encode_select(s.obs, ts, tables)
    finally:
        s.close()
    m = PolicyModel(tiny_config(tables))
    sb = collate_states([ts])
    trunk = m.encode(sb)
    selb = collate_selects([es])
    o = len(es.opt_type)
    picked = torch.zeros((1, o + 1), dtype=torch.bool)
    logits = m.option_logits(trunk, sb, selb, picked)
    assert logits.shape == (1, o + 1)
    if es.min_count >= 1:
        assert logits[0, o] == float("-inf")     # done illegal before min picks
    picked[0, 0] = True
    logits2 = m.option_logits(trunk, sb, selb, picked)
    assert logits2[0, 0] == float("-inf")        # picked option masked


def test_heads_shapes():
    from ptcg.model import PolicyModel, collate_states, tiny_config
    tables, states = _real_states(2)
    m = PolicyModel(tiny_config(tables))
    trunk = m.encode(collate_states(states))
    assert m.public_value(trunk).shape == (2,)
    assert m.public_value(trunk).abs().max() < 1.0
    assert m.aux_decklist(trunk).shape == (2, tables.n_rows)
    assert (m.aux_decklist(trunk) >= 0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_model.py -v -k "option_logits or heads"`
Expected: FAIL with `ImportError: cannot import name 'PolicyModel'`

- [ ] **Step 3: Implement** (append to `ptcg/model.py`):

```python
class PolicyModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None, attack_table=None):
        super().__init__()
        if attack_table is None:
            from .cards import build_tables
            t = build_tables()
            attr_table, attack_table = t.attr, t.attack_feat
        self.cfg = cfg
        self.encoder = Encoder(cfg, attr_table)
        self.register_buffer("atk", torch.from_numpy(attack_table))
        self.opt_type = nn.Embedding(F.N_OPT_TYPE, cfg.d)
        self.q_type = nn.Embedding(F.N_SELECT_TYPE, cfg.d)
        self.q_ctx = nn.Embedding(F.N_SELECT_CTX, cfg.d)
        self.atk_proj = nn.Linear(ATK_DIM, cfg.d)
        self.opt_scalar = nn.Linear(F.OPT_SCALAR_DIM, cfg.d)
        self.q_scalar = nn.Linear(F.Q_SCALAR_DIM, cfg.d)
        self.picked_proj = nn.Linear(cfg.d, cfg.d)
        self.done_tok = nn.Parameter(torch.zeros(cfg.d))
        self.opt_norm = nn.LayerNorm(cfg.d)
        dec = nn.TransformerDecoderLayer(
            cfg.d, cfg.heads, cfg.ffn, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec, cfg.dec_layers)
        self.logit = nn.Linear(cfg.d, 1)
        self.v_head = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(),
                                    nn.Linear(cfg.d, 1), nn.Tanh())
        self.pd_head = nn.Linear(2 * cfg.d, 1)
        self.dl_head = nn.Linear(cfg.d, cfg.n_card_rows)
        self.hd_head = nn.Linear(cfg.d, cfg.n_card_rows)

    def encode(self, state_batch):
        return self.encoder(state_batch)

    def _gather_ref(self, trunk, ref):
        safe = ref.clamp(min=0)
        g = torch.gather(trunk, 1, safe.unsqueeze(-1).expand(-1, -1, trunk.shape[-1]))
        return g * (ref >= 0).unsqueeze(-1)

    def option_logits(self, trunk, state_batch, sel, picked):
        B, O = sel["opt_type"].shape
        card_vec = self.encoder.card(sel["opt_card"])
        opt = (self.opt_type(sel["opt_type"]) + card_vec
               + self.atk_proj(self.atk[sel["opt_attack"]])
               + self.opt_scalar(sel["opt_scalar"])
               + self._gather_ref(trunk, sel["opt_ref"]))
        q = (self.q_type(sel["q_type"]) + self.q_ctx(sel["q_ctx"])
             + self.q_scalar(sel["q_scalar"])
             + self._gather_ref(trunk, sel["q_ref"]).sum(1))
        picked_sum = (opt * picked[:, :O].unsqueeze(-1)).sum(1)
        q = q + self.picked_proj(picked_sum)
        done = self.done_tok.expand(B, 1, -1)
        tgt = self.opt_norm(torch.cat([q.unsqueeze(1), opt, done], dim=1))
        h = self.decoder(tgt, trunk, memory_key_padding_mask=~state_batch["mask"])
        logits = self.logit(h[:, 1:, :]).squeeze(-1)          # [B, O+1]
        neg = torch.finfo(logits.dtype).min
        pad = ~sel["opt_mask"]
        logits[:, :O] = logits[:, :O].masked_fill(pad | picked[:, :O], neg)
        n_picked = picked[:, :O].sum(-1)
        done_illegal = n_picked < sel["min_count_t"]
        logits[:, O] = logits[:, O].masked_fill(done_illegal, neg)
        return logits

    def _pooled(self, trunk):
        return torch.cat([trunk[:, VALUE_ROWS[0]], trunk[:, VALUE_ROWS[1]]], dim=-1)

    def public_value(self, trunk):
        return self.v_head(self._pooled(trunk)).squeeze(-1)

    def prize_diff(self, trunk):
        return self.pd_head(self._pooled(trunk)).squeeze(-1)

    def aux_decklist(self, trunk):
        return nn.functional.softplus(self.dl_head(trunk[:, 0]))

    def aux_hand(self, trunk):
        return nn.functional.softplus(self.hd_head(trunk[:, 0]))


def collate_selects(selects, device=None):
    B = len(selects)
    O = max(len(s.opt_type) for s in selects)
    out = {
        "opt_type": torch.zeros(B, O, dtype=torch.int64),
        "opt_ref": torch.full((B, O), -1, dtype=torch.int64),
        "opt_card": torch.zeros(B, O, dtype=torch.int64),
        "opt_attack": torch.zeros(B, O, dtype=torch.int64),
        "opt_scalar": torch.zeros(B, O, F.OPT_SCALAR_DIM),
        "opt_mask": torch.zeros(B, O, dtype=torch.bool),
        "q_type": torch.zeros(B, dtype=torch.int64),
        "q_ctx": torch.zeros(B, dtype=torch.int64),
        "q_scalar": torch.zeros(B, F.Q_SCALAR_DIM),
        "q_ref": torch.full((B, 2), -1, dtype=torch.int64),
        "min_count_t": torch.zeros(B, dtype=torch.int64),
        "max_count_t": torch.zeros(B, dtype=torch.int64),
    }
    for i, s in enumerate(selects):
        o = len(s.opt_type)
        for k, arr in (("opt_type", s.opt_type), ("opt_ref", s.opt_ref),
                       ("opt_card", s.opt_card), ("opt_attack", s.opt_attack)):
            out[k][i, :o] = torch.as_tensor(arr)
        out["opt_scalar"][i, :o] = torch.as_tensor(s.opt_scalar)
        out["opt_mask"][i, :o] = True
        out["q_type"][i], out["q_ctx"][i] = s.q_type, s.q_ctx
        out["q_scalar"][i] = torch.as_tensor(s.q_scalar)
        out["q_ref"][i] = torch.as_tensor(s.q_ref)
        out["min_count_t"][i], out["max_count_t"][i] = s.min_count, s.max_count
    return out
```

- [ ] **Step 4: Run all model tests**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/model.py tests/test_model.py
git commit -m "pointer decoder, value and aux heads"
```

---

### Task 9: Composite action — shared sample/replay pick loop

**Files:**
- Create: `ptcg/action.py`
- Test: `tests/test_action.py`

**Interfaces:**
- Consumes: `PolicyModel.option_logits`, `collate_states`, `collate_selects`.
- Produces:

```python
@dataclass
class SelectDecision:
    picks: list[int]        # option indices, order as sampled (never includes done)
    logprob: float
    entropy: float

def run_pick_loop(model, trunk, state_batch, sel_batch, *,
                  forced: list[int] | None = None,
                  generator: torch.Generator | None = None,
                  ) -> tuple[list[int], Tensor, Tensor]
    # single-sample (B==1). Returns (picks, logprob_tensor, entropy_tensor).
    # forced=None: sample. forced=picks: force those picks in order, then force
    # done iff len(forced) < max_count. logprob_tensor is differentiable.

def sample_select(model, ts, es, generator) -> SelectDecision
def replay_logprob(model, states_b, sels_b, picks_list) -> Tensor  # [B], batched replay
```

Loop semantics (both paths run this exact function): at each step build
`picked` mask, get `option_logits`, `Categorical(logits=...)`; sample or take
forced action; accumulate `log_prob` and `entropy`; stop on done action or when
`len(picks) == max_count`. Replay forces a done step only when the stored picks
ended before `max_count`.

- [ ] **Step 1: Write the failing test** — `tests/test_action.py`:

```python
import random
import torch
from ptcg.action import replay_logprob, sample_select
from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck
from ptcg.featurize import encode_select, featurize_state
from ptcg.model import PolicyModel, tiny_config
from ptcg.tracker import BeliefTracker


def _one_select():
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        me = s.select_player
        ts = featurize_state(s.obs, me, deck, BeliefTracker(me).snapshot(), tables)
        es = encode_select(s.obs, ts, tables)
    finally:
        s.close()
    return tables, ts, es


def test_sample_respects_bounds_and_no_duplicates():
    tables, ts, es = _one_select()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        d = sample_select(m, ts, es, g)
        assert es.min_count <= len(d.picks) <= es.max_count
        assert len(set(d.picks)) == len(d.picks)
        assert all(0 <= p < len(es.opt_type) for p in d.picks)


def test_replay_matches_sample_exactly():
    tables, ts, es = _one_select()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    d = sample_select(m, ts, es, g)
    lp = replay_logprob(m, [ts], [es], [d.picks])
    assert torch.allclose(lp[0], torch.tensor(d.logprob), atol=0, rtol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_action.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.action'`

- [ ] **Step 3: Implement `ptcg/action.py`**

```python
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .model import collate_selects, collate_states


@dataclass
class SelectDecision:
    picks: list
    logprob: float
    entropy: float


def run_pick_loop(model, trunk, state_batch, sel_batch, *, forced=None, generator=None):
    O = sel_batch["opt_type"].shape[1]
    max_count = int(sel_batch["max_count_t"][0])
    picked = torch.zeros((1, O + 1), dtype=torch.bool)
    picks, logp, ent = [], trunk.new_zeros(()), trunk.new_zeros(())
    step = 0
    while True:
        logits = model.option_logits(trunk, state_batch, sel_batch, picked)
        dist = Categorical(logits=logits)
        if forced is None:
            a = dist.sample() if generator is None else torch.multinomial(
                dist.probs.squeeze(0), 1, generator=generator)
            a = a.reshape(1)
        else:
            a = torch.tensor([forced[step] if step < len(forced) else O])
        logp = logp + dist.log_prob(a).squeeze(0)
        ent = ent + dist.entropy().squeeze(0)
        if int(a) == O:  # done
            break
        picks.append(int(a))
        picked[0, int(a)] = True
        step += 1
        if len(picks) == max_count:
            break
    return picks, logp, ent


def sample_select(model, ts, es, generator=None) -> SelectDecision:
    sb = collate_states([ts])
    selb = collate_selects([es])
    with torch.no_grad():
        trunk = model.encode(sb)
        picks, logp, ent = run_pick_loop(model, trunk, sb, selb, generator=generator)
    return SelectDecision(picks, float(logp), float(ent))


def replay_logprob(model, states, sels, picks_list):
    out = []
    for ts, es, picks in zip(states, sels, picks_list):
        sb = collate_states([ts])
        selb = collate_selects([es])
        trunk = model.encode(sb)
        _, logp, _ = run_pick_loop(model, trunk, sb, selb, forced=list(picks))
        out.append(logp)
    return torch.stack(out)
```

(Batched-trunk replay is a later optimization for the training plan; the contract
here is exactness, and both paths share `run_pick_loop`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_action.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/action.py tests/test_action.py
git commit -m "shared sample and replay pick loop"
```

---

### Task 10: Privileged critic module

**Files:**
- Modify: `ptcg/model.py` (append), `tests/test_model.py` (append)

**Interfaces:**
- Produces:

```python
def critic_config(tables) -> ModelConfig       # d256 L4 H8 ffn1024
class CriticModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None)
    def forward(self, priv_state_batch) -> Tensor   # [B, 2] in (-1,1): value for p0, p1
```

- [ ] **Step 1: Write the failing test** (append to `tests/test_model.py`):

```python
def test_critic_two_player_values():
    from ptcg.model import CriticModel, collate_states, critic_config
    tables, states = _real_states(2)   # public states stand in; shapes identical
    c = CriticModel(critic_config(tables))
    v = c(collate_states(states))
    assert v.shape == (2, 2)
    assert v.abs().max() < 1.0
```

- [ ] **Step 2: Run to verify failure, implement, re-run**

Run: `python3 -m pytest tests/test_model.py -v -k critic` → FAIL (`ImportError`).

```python
def critic_config(tables):
    return ModelConfig(d=256, layers=4, heads=8, ffn=1024, dec_layers=0,
                       n_card_rows=tables.n_rows, n_attack_rows=tables.n_attack_rows)


class CriticModel(nn.Module):
    def __init__(self, cfg: ModelConfig, attr_table=None):
        super().__init__()
        self.encoder = Encoder(cfg, attr_table)
        self.head = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(),
                                  nn.Linear(cfg.d, 2), nn.Tanh())

    def forward(self, batch):
        h = self.encoder(batch)
        return self.head(torch.cat([h[:, VALUE_ROWS[0]], h[:, VALUE_ROWS[1]]], -1))
```

Re-run → PASS.

- [ ] **Step 3: Commit**

```bash
git add ptcg/model.py tests/test_model.py
git commit -m "privileged critic model"
```

---

### Task 11: Rollout — model plays legal full games

**Files:**
- Create: `ptcg/rollout.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: everything above.
- Produces:

```python
@dataclass
class Step:
    player: int
    state: TokenizedState
    esel: EncodedSelect
    picks: list[int]
    logprob: float
    priv_state: TokenizedState   # privileged view at this step

@dataclass
class Episode:
    steps: list[Step]
    result: int                  # 0/1 winner, 2 draw
    rewards: tuple[float, float] # (+1,-1) / (-1,+1) / (0,0)

def play_game(model, decks: tuple[list[int], list[int]], tables,
              generator=None, step_cap: int = 5000) -> Episode
```

`play_game` runs both seats with the same model (self-play), one `BeliefTracker`
per seat updated with that seat's logs before featurizing, builds the privileged
state from the two seats' latest observations, and raises if the engine rejects a
selection (`IndexError` must propagate — legality is the whole point).

- [ ] **Step 1: Write the failing test** — `tests/test_integration.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_game


def test_untrained_model_plays_legal_games():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        ep = play_game(m, (deck, list(deck)), tables, generator=g)
        assert ep.result in (0, 1, 2)
        assert 5 <= len(ep.steps) <= 1000
        assert all(s.logprob <= 0 and s.logprob == s.logprob for s in ep.steps)
        assert ep.rewards in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.rollout'`

- [ ] **Step 3: Implement `ptcg/rollout.py`**

```python
from dataclasses import dataclass

from .action import SelectDecision, sample_select
from .engine import BattleSession
from .featurize import (EncodedSelect, TokenizedState, encode_select,
                        featurize_privileged, featurize_state)
from .tracker import BeliefTracker


@dataclass
class Step:
    player: int
    state: TokenizedState
    esel: EncodedSelect
    picks: list
    logprob: float
    priv_state: TokenizedState


@dataclass
class Episode:
    steps: list
    result: int
    rewards: tuple


def play_game(model, decks, tables, generator=None, step_cap=5000):
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    last_obs = [s.obs, s.obs]
    steps = []
    try:
        while not s.done:
            if len(steps) >= step_cap:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            last_obs[me] = s.obs
            trackers[me].update(s.obs.get("logs", []))
            ts = featurize_state(s.obs, me, decks[me], trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            pv = featurize_privileged(last_obs[0], last_obs[1], decks, tables)
            d = sample_select(model, ts, es, generator)
            steps.append(Step(me, ts, es, d.picks, d.logprob, pv))
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    return Episode(steps, r, rewards)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_integration.py -v`
Expected: PASS (5 games with tiny model; expect ~1–3 minutes on CPU)

- [ ] **Step 5: Commit**

```bash
git add ptcg/rollout.py tests/test_integration.py
git commit -m "self-play rollout with legality proof"
```

---

### Task 12: PPO-shaped gradient smoke test (ratio ≡ 1 contract)

**Files:**
- Test: `tests/test_ppo_smoke.py` (test-only task; any fixes it forces happen in
  the modules that fail it)

**Interfaces:**
- Consumes: `Episode`/`Step`, `replay_logprob`, `CriticModel`, `PolicyModel`
  heads.

- [ ] **Step 1: Write the test**

```python
import torch
from ptcg.action import replay_logprob
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import (CriticModel, PolicyModel, collate_states,
                        critic_config, tiny_config)
from ptcg.rollout import play_game


def _gae(deltas, gamma=1.0, lam=0.95):
    adv, out = 0.0, []
    for d in reversed(deltas):
        adv = d + gamma * lam * adv
        out.append(adv)
    return list(reversed(out))


def test_ratio_is_one_and_gradients_flow():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    g = torch.Generator().manual_seed(2)
    eps = [play_game(m, (deck, list(deck)), tables, generator=g) for _ in range(2)]

    steps = [s for ep in eps for s in ep.steps]
    old_lp = torch.tensor([s.logprob for s in steps])
    new_lp = replay_logprob(m, [s.state for s in steps],
                            [s.esel for s in steps], [s.picks for s in steps])
    ratio = (new_lp - old_lp).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), \
        "shared pick loop must reproduce actor log-probs exactly"

    # per-seat advantages from the privileged critic, gamma=1 terminal reward
    with torch.no_grad():
        pv = critic(collate_states([s.priv_state for s in steps]))
    adv, ret = [], []
    off = 0
    for ep in eps:
        idx = list(range(off, off + len(ep.steps)))
        off += len(ep.steps)
        for seat in (0, 1):
            rows = [i for i in idx if steps[i].player == seat]
            vals = [float(pv[i, seat]) for i in rows]
            rw = ep.rewards[seat]
            deltas = [
                (vals[j + 1] if j + 1 < len(vals) else rw) - vals[j]
                for j in range(len(vals))
            ]
            a = _gae(deltas)
            adv += [(rows[j], a[j]) for j in range(len(rows))]
            ret += [(rows[j], a[j] + vals[j]) for j in range(len(rows))]
    order = [i for i, _ in sorted(adv)]
    advt = torch.tensor([a for _, a in sorted(adv)])
    rett = torch.tensor([r for _, r in sorted(ret)])

    opt = torch.optim.Adam(list(m.parameters()) + list(critic.parameters()), lr=3e-4)
    losses = []
    for _ in range(3):
        new_lp = replay_logprob(m, [steps[i].state for i in order],
                                [steps[i].esel for i in order],
                                [steps[i].picks for i in order])
        ratio = (new_lp - old_lp[order]).exp()
        clipped = torch.clamp(ratio, 0.8, 1.2)
        pg = -torch.min(ratio * advt, clipped * advt).mean()
        trunkb = collate_states([steps[i].state for i in order])
        v = m.public_value(m.encode(trunkb))
        vloss = ((v - rett) ** 2).mean()
        pvb = critic(collate_states([steps[i].priv_state for i in order]))
        closs = ((pvb[range(len(order)), [steps[i].player for i in order]] - rett) ** 2).mean()
        loss = pg + 0.5 * vloss + 0.5 * closs
        opt.zero_grad()
        loss.backward()
        for name, p in list(m.named_parameters()) + list(critic.named_parameters()):
            assert p.grad is None or torch.isfinite(p.grad).all(), name
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], "loss should decrease when overfitting one batch"
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_ppo_smoke.py -v`
Expected: PASS. If the ratio assertion fails, the bug is in
`run_pick_loop`/`option_logits` determinism (e.g., dropout active — models must
be constructed in eval-compatible form; `nn.TransformerEncoderLayer` defaults to
dropout=0.1, so config must set dropout=0.0 — if this bites, add
`dropout: float = 0.0` to `ModelConfig`, pass it through both layer constructors
in Tasks 7/8/10, and re-run Tasks 7–12 tests).

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ppo_smoke.py
git commit -m "ppo smoke test: ratio-one contract and gradient flow"
```

---

### Task 13: Featurizer throughput microbenchmark

**Files:**
- Create: `benchmarks/bench_featurizer.py`, `benchmarks/RESULTS-featurizer.md`

**Interfaces:**
- Consumes: `featurize_state`, `encode_select`, `BattleSession`, `random_picks`.
- Produces: measured µs per (featurize_state + encode_select) call, appended to
  a short results file — this is the number the training-pipeline spec's
  throughput budget needs (env step alone is ~297 µs; the featurizer should not
  multiply it).

- [ ] **Step 1: Write the benchmark** (same conventions as existing
  `benchmarks/` scripts: stdlib + project imports, `time.perf_counter`, warmup,
  ≥30 s window):

```python
"""Featurizer throughput: µs per featurize_state+encode_select over random games."""
import random
import statistics
import time

from ptcg.cards import build_tables
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import encode_select, featurize_state
from ptcg.tracker import BeliefTracker

tables = build_tables()
deck = load_sample_deck()
rng = random.Random(0)
per_call = []
t_end = time.perf_counter() + 30
games = sel = 0
while time.perf_counter() < t_end:
    s = BattleSession(deck, list(deck))
    trackers = [BeliefTracker(0), BeliefTracker(1)]
    try:
        while not s.done:
            me = s.select_player
            trackers[me].update(s.obs.get("logs", []))
            t0 = time.perf_counter()
            ts = featurize_state(s.obs, me, deck, trackers[me].snapshot(), tables)
            encode_select(s.obs, ts, tables)
            per_call.append(time.perf_counter() - t0)
            sel += 1
            s.select(random_picks(s.obs, rng))
    finally:
        s.close()
    games += 1
us = [x * 1e6 for x in per_call]
print(f"games={games} selections={sel}")
print(f"featurize+encode: mean {statistics.mean(us):.0f} us, "
      f"median {statistics.median(us):.0f} us, "
      f"p95 {sorted(us)[int(0.95 * len(us))]:.0f} us")
```

- [ ] **Step 2: Run and record**

Run: `python3 benchmarks/bench_featurizer.py`
Write the three output lines plus hardware note and date into
`benchmarks/RESULTS-featurizer.md`. No acceptance threshold — this is
measurement; if mean exceeds ~600 µs (2× the env step), add a one-line "needs
optimization in training plan" note.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_featurizer.py benchmarks/RESULTS-featurizer.md
git commit -m "featurizer throughput benchmark"
```

---

## Self-review notes (kept for the record)

- Spec coverage: tokenization → Tasks 4–6; trunk → 7; decoder/pointer/values/aux
  → 8; composite action + shared pick loop + ratio contract → 9 and 12;
  privileged critic → 6 and 10; RL interface (per-select timestep, γ=1 GAE) → 12;
  featurizer-throughput open question → 13. Deferred per Global Constraints:
  text-encoder init, archetype head, trunk caching, plan tokens (the latter two
  are spec-deferred already).
- The dropout hazard (torch layers default to 0.1, which would break both the
  padding-invariance test and the ratio contract) is called out in Task 12 with
  the exact fix path.
- Type consistency: `TokenizedState`/`EncodedSelect` field names match between
  Tasks 4/5 (producers) and 7–12 (consumers); `min_count_t`/`max_count_t` are the
  collated tensor names, `min_count`/`max_count` the per-sample ints.
