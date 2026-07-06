# PTCG league + deck portfolio (phase 3) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-deck, population-based (league) self-play training that makes the
policy robust across a portfolio of real metagame decks and a pool of past
opponents, per `docs/superpowers/specs/2026-07-05-league-deck-portfolio-phase3-design.md`.

**Architecture:** Extends the phase-2 round loop. Each game samples a deck per
side from a curated portfolio and an opponent policy from a {current, past
snapshot, random} mixture; the learner plays the current policy and collects
only its own seat's on-policy steps (both seats for mirror games). A league pool
of policy-only snapshots grows every few rounds. All phase-2 numerical machinery
(batched replay, PPO, ratio-one gate, GAE, resume) is reused unchanged and only
ever runs over collected learner-seat steps.

**Tech Stack:** the existing `ptcg` package; Python 3.12 training venv
(`venv-train`, torch 2.5.1+cu121, device cuda, `--minibatch 128`); base 3.14
interpreter for CPU tests; numpy, pytest, matplotlib.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-05-league-deck-portfolio-phase3-design.md`.
  On conflict the spec wins; deviations go in the final report.
- Reuse phase-2 contracts verbatim: shared pick loop / batched replay with the
  ratio-one contract (epoch-0 `max|ratio−1| < 1e-3` abort); dropout 0.0; option
  list is the only legality oracle; pure featurizers; one battle per process;
  fp32 on the 1060 with `--minibatch 128`; checkpoint IO/resume with
  models-to-device-before-optimizer-restore. Never modify anything under
  `pokemon-tcg-ai-battle/`.
- The engine is the deck legality oracle: a deck is legal iff
  `g.battle_start(deck0, deck1)` returns a non-None obs (else it returns
  `(None, start)` with `start.errorPlayer`/`start.errorType`). `BattleSession`
  raises `ValueError` on rejection — catch it to classify legality.
- Correctness-critical rule (spec §Data collection): against a past-snapshot or
  random opponent, collect ONLY the learner seat's steps; for mirror games
  (opponent = current policy) collect both. Opponent-seat steps never enter the
  training buffer or replay.
- All CPU tests use `tiny_config` and run as ONE pytest invocation (never
  parallel; one battle per process). Engine-dependent tests assert invariants
  and ranges, never exact RNG outcomes.
- Commit style: short lowercase messages, directly on `main`. Windows: `python`
  (tests) / `venv-train\Scripts\python` (training); multiprocessing spawn; every
  process-entry function top-level; `if __name__ == "__main__"` on scripts.
- Long training launches must be fully detached (never a child of a transient
  session) — a phase-2 lesson.

## Existing phase-2 interfaces this plan builds on (do not change their behavior)

```python
# ptcg/rollout.py
@dataclass
class Step: player:int; state:TokenizedState; esel:EncodedSelect; picks:list; logprob:float; priv_state:TokenizedState
@dataclass
class Episode: steps:list; result:int; rewards:tuple; featurizer_version:int=FEATURIZER_VERSION
def play_game(model, decks, tables, generator=None, step_cap=5000, priv_viz=False, obs_log=None) -> Episode
# ptcg/actors.py
class _NullCritic  # .load_state_dict no-op
def collect_round_worker(args) -> dict          # args=(cfg_json,round,actor,n_games,ckpt_path)
def run_actor_pool(cfg, round_n, ckpt_path, worker=collect_round_worker, extra=None) -> list[dict]
def play_versus(model, opponent, tables, decks, generator, model_seat, step_cap=5000) -> int  # 1 iff model_seat won
def eval_worker(args) -> dict                   # args=(cfg_json,round,actor,n_games,ckpt_path,opp_spec)
# ptcg/trainloop.py
@dataclass class TrainConfig  # fields incl. run_dir, model_size, games_per_round, actors, device, seed, eval_every, minibatch, step_cap, ...
def model_config_for(size, tables); round_dir(cfg,n)->Path; checkpoint_path(cfg,n)->Path
def game_seed(cfg, round_n, actor_idx, game_idx)->int; save_game(path, episode); load_round(cfg,n)->list[Episode]
def save_checkpoint(cfg,n,policy,critic,optim); load_checkpoint(path,policy,critic,optim=None)->int
def latest_checkpoint(cfg); prune_checkpoints(cfg,current_round)->list[str]
def learner_update(policy, critic, optim, episodes, cfg, tables, opp_deck)->dict
def append_metrics(cfg,row); read_metrics(cfg)->list[dict]; truncate_metrics(cfg,before_round)
def _eval_due(cfg, round_n, policy, tables)->dict|None; def train(cfg, max_rounds)
METRIC_FIELDS = [...]
# ptcg/ppo.py
def assemble_advantages(episodes, critic, device=None, lam=.95, gamma=1., normalize=True) -> (steps, old_lp, adv, ret)
def aux_targets(steps, tables, opp_deck) -> (pd, dl, hd)
def ppo_policy_loss(new_lp, old_lp, adv, clip=.2) -> (pg, ratio, approx_kl)
# ptcg/model.py: PolicyModel, CriticModel, student_config/tiny_config/critic_config, collate_states, collate_selects
# ptcg/cards.py: build_tables()->CardTables (.n_rows, .attr), card_row(cid, n_rows)
# ptcg/engine.py: BattleSession(deck0,deck1), load_sample_deck()->list[int], random_picks(obs,rng)
```

## File Structure

```
ptcg/decks.py       — name→ID resolver, curated decklists (data), two-gate validate, PORTFOLIO (Tasks 1-2)
ptcg/league.py      — policy-only snapshot store, mixture sampler, capped+anchored retention (Task 3)
ptcg/rollout.py     — MODIFY: add play_league_game + extend Episode (Task 4)
ptcg/ppo.py         — MODIFY: aux_targets accepts per-step opp_decks (Task 5)
ptcg/actors.py      — MODIFY: league_round_worker; eval_worker per-deck + vs-SD-champ (Tasks 6, 8)
ptcg/trainloop.py   — MODIFY: TrainConfig league fields; league snapshotting; train() wiring (Task 7)
scripts/build_decks.py     — source/validate/commit the portfolio (Task 2)
scripts/train.py, plot_run.py — MODIFY: flags auto-extend; new curves (Task 9)
docs/phase3-run.md  — qualification-run playbook (Task 10)
tests/test_decks.py … tests/test_league_integration.py — one per task
```

---

### Task 1: Deck name→ID resolver + two-gate validation

**Files:**
- Create: `ptcg/decks.py`
- Test: `tests/test_decks.py`

**Interfaces:**
- Consumes: `ptcg.engine` (`engine_dir`, `BattleSession`, `random_picks`,
  `load_sample_deck`), `pokemon-tcg-ai-battle/EN_Card_Data.csv`.
- Produces:

```python
def card_name_index() -> dict[str, int]
    # normalized card-name -> a canonical card ID (first ID for that name),
    # built from EN_Card_Data.csv column "Card Name"/"Card ID". Cached.
def resolve(name: str) -> int | None          # normalized lookup; None if unknown
def deck_from_counts(counts: list[tuple[str, int]]) -> list[int]
    # [("Charizard ex", 3), ("Rare Candy", 4), ...] -> flat 60-int deck via resolve
def is_legal(deck: list[int]) -> tuple[bool, str]
    # engine gate: True,"" if battle_start accepts deck vs itself; else False,reason
def is_playable(deck: list[int], n_games: int = 6, seed: int = 0) -> tuple[bool, str]
    # random-vs-random for n_games; True iff all complete within step cap AND at
    # least one win for each seat across the games (not a degenerate deck)
def validate(deck: list[int]) -> tuple[bool, str]   # is_legal AND is_playable
```

- Normalization: lowercase, strip, collapse internal whitespace, drop trailing
  set/number parentheticals so `"Charizard ex (OBF 125)"` → `"charizard ex"`.

- [ ] **Step 1: Write the failing test** — `tests/test_decks.py`:

```python
from ptcg.decks import (card_name_index, resolve, deck_from_counts,
                        is_legal, is_playable, validate)
from ptcg.engine import load_sample_deck


def test_name_index_and_resolve():
    idx = card_name_index()
    assert len(idx) > 500                    # real card pool
    # sample deck is Gardevoir line; those names must resolve
    assert resolve("Gardevoir ex") is not None
    assert resolve("  gardevoir EX  ") == resolve("Gardevoir ex")  # normalized
    assert resolve("Not A Real Card 9999") is None


def test_sample_deck_is_legal_and_playable():
    deck = load_sample_deck()
    ok, why = is_legal(deck)
    assert ok, why
    ok, why = is_playable(deck, n_games=4, seed=1)
    assert ok, why
    assert validate(deck)[0]


def test_illegal_deck_rejected():
    # 60 copies of one non-energy card is not a legal deck
    bad = [resolve("Gardevoir ex")] * 60
    ok, why = is_legal(bad)
    assert not ok and why  # engine rejects; reason non-empty


def test_deck_from_counts_builds_60():
    deck = load_sample_deck()
    # round-trip a known-good structure: rebuild the sample deck from its counts
    from collections import Counter
    # deck_from_counts needs names; use ids directly via a numeric passthrough
    # (deck_from_counts accepts already-int "names" too — see impl)
    counts = [(str(c), n) for c, n in Counter(deck).items()]
    rebuilt = deck_from_counts(counts)
    assert len(rebuilt) == 60 and Counter(rebuilt) == Counter(deck)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_decks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.decks'`

- [ ] **Step 3: Implement `ptcg/decks.py`**

```python
import csv
import os
import random
from functools import lru_cache

from .engine import BattleSession, engine_dir, random_picks


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    if "(" in s:
        s = s[: s.index("(")].strip()
    return " ".join(s.split())


@lru_cache(maxsize=1)
def card_name_index() -> dict:
    path = os.path.join(os.path.dirname(engine_dir()), "..", "EN_Card_Data.csv")
    path = os.path.normpath(path)
    idx = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = _norm(row["Card Name"])
            cid = int(row["Card ID"])
            idx.setdefault(key, cid)  # first ID for a given name (reprints share function)
    return idx


def resolve(name):
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        return int(name)
    return card_name_index().get(_norm(name))


def deck_from_counts(counts) -> list:
    deck = []
    for name, n in counts:
        cid = resolve(name)
        if cid is None:
            raise KeyError(f"unresolved card: {name!r}")
        deck.extend([cid] * n)
    return deck


def is_legal(deck):
    try:
        s = BattleSession(list(deck), list(deck))
    except ValueError as e:
        return False, str(e)
    else:
        s.close()
        return True, ""


def is_playable(deck, n_games: int = 6, seed: int = 0):
    seat_wins = [0, 0]
    rng = random.Random(seed)
    for _ in range(n_games):
        try:
            s = BattleSession(list(deck), list(deck))
        except ValueError as e:
            return False, f"illegal: {e}"
        try:
            n = 0
            while not s.done:
                n += 1
                if n > 5000:
                    return False, "did not terminate"
                s.select(random_picks(s.obs, rng))
            if s.result in (0, 1):
                seat_wins[s.result] += 1
        finally:
            s.close()
    if seat_wins[0] == 0 or seat_wins[1] == 0:
        return False, f"degenerate outcomes {seat_wins}"
    return True, ""


def validate(deck):
    ok, why = is_legal(deck)
    if not ok:
        return False, why
    return is_playable(deck)
```

Note: verify the `EN_Card_Data.csv` path resolves — `engine_dir()` points at
`pokemon-tcg-ai-battle/sample_submission/sample_submission`; the CSV is at
`pokemon-tcg-ai-battle/EN_Card_Data.csv` (two levels up). Adjust the `..`
count if the test's `card_name_index` length assertion fails with a
FileNotFoundError, and record the resolved path in your report.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_decks.py -v`
Expected: 4 PASS (a few seconds; `is_playable` plays real games).

- [ ] **Step 5: Commit**

```bash
git add ptcg/decks.py tests/test_decks.py
git commit -m "deck name-to-id resolver and two-gate validation"
```

---

### Task 2: Curated deck portfolio

**Files:**
- Modify: `ptcg/decks.py` (append `PORTFOLIO`, `SAMPLE`, `train_decks`, `all_decks`)
- Create: `scripts/build_decks.py`, `ptcg/decklists.py` (decklists as data)
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Produces:

```python
# ptcg/decks.py
SAMPLE = "gardevoir_sample"          # name of the phase-1 sample deck entry
PORTFOLIO: dict[str, list[int]]      # name -> validated 60-int deck (>=6 entries incl. SAMPLE)
def all_decks() -> list[str]         # portfolio names
def train_decks() -> list[str]       # names used in training (all of them in phase 3)
def deck(name: str) -> list[int]     # PORTFOLIO[name]
```

- `ptcg/decklists.py` holds `RAW: dict[str, list[tuple[str,int]]]` — human/
  meta decklists as (card-name, count) pairs (the sample deck as literal ids).
  `PORTFOLIO` is built by resolving+validating each `RAW` entry at import;
  entries that fail validation are omitted with a module-level
  `SKIPPED: dict[str,str]` recording why.

- [ ] **Step 1: Source the decklists** (`scripts/build_decks.py`)

This step gathers real current-Standard metagame decklists and writes
`ptcg/decklists.py`. It is a data-curation step, not TDD. Procedure:

1. Sourcing (in the training venv, web tools available to the implementer, or
   from user-provided `deck.csv` / decklist text files dropped in `decks_src/`):
   gather 6–10 distinct top-archetype lists (Charizard ex, Dragapult ex, Raging
   Bolt ex, Gardevoir ex, Miraidon ex, Lugia VSTAR, Gholdengo ex, a Lost-Zone
   toolbox — pick what's current and mappable), each as (card-name, count) pairs
   summing to 60. The phase-1 sample deck is `SAMPLE`, stored as literal card
   ids (from `load_sample_deck()`), not names.
2. For each candidate, resolve names → ids (`deck_from_counts`), then
   `validate()`. Print a table: name, resolved/60, legal?, playable?, skip
   reason. Repair unresolved cards (alternate name spellings) or drop the deck.
3. Write `ptcg/decklists.py` with `RAW` = the validated (name,count) lists, and
   a comment noting the source and date of each list.

`scripts/build_decks.py` skeleton:

```python
"""Source, resolve, validate, and emit the deck portfolio. Run in venv-train.
Writes/updates ptcg/decklists.py. Not a test; run once when curating decks."""
from ptcg.decks import deck_from_counts, validate, resolve
from ptcg.engine import load_sample_deck

# (card-name, count) lists gathered from current Standard meta sources.
CANDIDATES = {
    # "charizard_ex": [("Charmander", 3), ("Charmeleon", 1), ("Charizard ex", 3), ...],
    # ... filled during curation ...
}


def main():
    ok = {}
    for name, counts in CANDIDATES.items():
        try:
            d = deck_from_counts(counts)
        except KeyError as e:
            print(f"{name:20} UNRESOLVED {e}")
            continue
        good, why = validate(d)
        print(f"{name:20} {'OK' if good else 'SKIP'} {why}")
        if good:
            ok[name] = counts
    # emit ptcg/decklists.py
    lines = ['"""Curated Standard-meta decklists (name,count). Built by scripts/build_decks.py."""',
             "SAMPLE_IDS = " + repr(load_sample_deck()), "", "RAW = {"]
    for name, counts in ok.items():
        lines.append(f"    {name!r}: {counts!r},")
    lines += ["}", ""]
    open("ptcg/decklists.py", "w").write("\n".join(lines))
    print(f"wrote {len(ok)} decks to ptcg/decklists.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing test** — `tests/test_portfolio.py`:

```python
from collections import Counter
from ptcg.decks import PORTFOLIO, SAMPLE, all_decks, train_decks, deck, is_legal


def test_portfolio_has_diverse_validated_decks():
    assert SAMPLE in PORTFOLIO
    assert len(PORTFOLIO) >= 6, PORTFOLIO.keys()
    for name in all_decks():
        d = deck(name)
        assert len(d) == 60, (name, len(d))
        ok, why = is_legal(d)
        assert ok, (name, why)          # every committed deck is engine-legal
    # decks are actually distinct (not the same list under many names)
    sigs = {name: tuple(sorted(Counter(deck(name)).items())) for name in all_decks()}
    assert len(set(sigs.values())) == len(sigs), "portfolio decks not distinct"


def test_train_decks_cover_portfolio():
    assert set(train_decks()) == set(all_decks())   # phase 3: all decks train
```

- [ ] **Step 3: Implement `PORTFOLIO` in `ptcg/decks.py`** (append)

```python
SAMPLE = "gardevoir_sample"


@lru_cache(maxsize=1)
def _build_portfolio():
    from . import decklists
    out = {SAMPLE: list(decklists.SAMPLE_IDS)}
    skipped = {}
    for name, counts in decklists.RAW.items():
        try:
            d = deck_from_counts(counts)
        except KeyError as e:
            skipped[name] = f"unresolved {e}"
            continue
        ok, why = validate(d)
        (out.__setitem__(name, d) if ok else skipped.__setitem__(name, why))
    return out, skipped


PORTFOLIO = _build_portfolio()[0]
SKIPPED = _build_portfolio()[1]


def all_decks():
    return list(PORTFOLIO)


def train_decks():
    return list(PORTFOLIO)


def deck(name):
    return list(PORTFOLIO[name])
```

- [ ] **Step 4: Run build + tests**

Run: `venv-train\Scripts\python scripts\build_decks.py` (curate until ≥6 decks
validate), then `python -m pytest tests/test_portfolio.py -v` → 2 PASS.
If fewer than 6 real decks map cleanly, fill the gap with programmatically
assembled decks (spec fallback) added to `CANDIDATES` until the count is met;
record which decks are real-meta vs assembled in your report.

- [ ] **Step 5: Commit**

```bash
git add ptcg/decks.py ptcg/decklists.py scripts/build_decks.py tests/test_portfolio.py
git commit -m "curated deck portfolio"
```

---

### Task 3: League snapshot store + mixture sampler

**Files:**
- Create: `ptcg/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `PolicyModel`, `model_config_for`, `_NullCritic`, `load_checkpoint`;
  `torch.save`/`load`.
- Produces:

```python
def league_dir(cfg) -> Path                       # <run_dir>/league
def snapshot(cfg, round_n, policy) -> Path        # save policy-only weights -> league/snap-<round>.pt (atomic)
def snapshot_rounds(cfg) -> list[int]             # sorted rounds present in the pool
def prune_pool(cfg, cap, anchors=2) -> list[int]  # keep `anchors` earliest + newest up to cap; return kept rounds
def sample_opponent(cfg, round_n, rng) -> tuple[str, str | None]
    # returns ("current", None) | ("pool", "<path>") | ("random", None)
    # by cfg.mirror_frac / cfg.pool_frac / cfg.random_frac; pool draws uniformly
    # from snapshot_rounds; if pool empty, falls back to "current".
def load_opponent(path, tables, cfg) -> PolicyModel    # policy-only, eval()
```

- [ ] **Step 1: Write the failing test** — `tests/test_league.py`:

```python
import random
import torch
from ptcg.cards import build_tables
from ptcg.league import (league_dir, snapshot, snapshot_rounds, prune_pool,
                        sample_opponent, load_opponent)
from ptcg.model import PolicyModel, tiny_config
from ptcg.trainloop import TrainConfig


def _cfg(tmp_path, **kw):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny", **kw)


def test_snapshot_and_prune(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path)
    p = PolicyModel(tiny_config(tables))
    for r in (0, 5, 10, 15, 20, 25):
        snapshot(cfg, r, p)
    assert snapshot_rounds(cfg) == [0, 5, 10, 15, 20, 25]
    kept = prune_pool(cfg, cap=4, anchors=2)
    # 2 earliest anchors (0,5) + newest up to cap: total 4 -> {0,5,20,25}
    assert kept == [0, 5, 20, 25]
    assert snapshot_rounds(cfg) == [0, 5, 20, 25]


def test_snapshot_roundtrip_loads_same_weights(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path)
    p = PolicyModel(tiny_config(tables))
    path = snapshot(cfg, 3, p)
    q = load_opponent(str(path), tables, cfg)
    for a, b in zip(p.state_dict().values(), q.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())


def test_sample_opponent_mixture(tmp_path):
    tables = build_tables()
    cfg = _cfg(tmp_path, mirror_frac=0.0, pool_frac=1.0, random_frac=0.0)
    p = PolicyModel(tiny_config(tables))
    snapshot(cfg, 0, p)
    kinds = [sample_opponent(cfg, 5, random.Random(i))[0] for i in range(10)]
    assert set(kinds) == {"pool"}          # pool_frac=1 always draws pool
    cfg2 = _cfg(tmp_path, mirror_frac=1.0, pool_frac=0.0, random_frac=0.0)
    assert sample_opponent(cfg2, 5, random.Random(0)) == ("current", None)
    # empty pool falls back to current even with pool_frac=1
    cfg3 = _cfg(str(tmp_path) + "_empty", mirror_frac=0.0, pool_frac=1.0,
                random_frac=0.0)
    assert sample_opponent(cfg3, 5, random.Random(0))[0] == "current"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.league'`
(and `TrainConfig` has no `mirror_frac` — Task 7 adds the fields; for THIS task
add them to TrainConfig first as part of your commit, or the test's `_cfg`
kwargs will TypeError. Add `mirror_frac: float = 0.30`, `pool_frac: float =
0.65`, `random_frac: float = 0.05`, `snapshot_every: int = 5`, `pool_cap: int =
18`, `sd_champ_ckpt: str = ""` to `TrainConfig` now; Task 7 wires them into
train()).

- [ ] **Step 3: Implement `ptcg/league.py`** and the TrainConfig fields.

```python
import os
from pathlib import Path

import torch

from .model import PolicyModel
from .trainloop import model_config_for
from .actors import _NullCritic


def league_dir(cfg) -> Path:
    return Path(cfg.run_dir) / "league"


def _snap_path(cfg, r) -> Path:
    return league_dir(cfg) / f"snap-{r:04d}.pt"


def snapshot(cfg, round_n, policy) -> Path:
    d = league_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = _snap_path(cfg, round_n)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({k: v.cpu() for k, v in policy.state_dict().items()}, tmp)
    os.replace(tmp, path)
    return path


def snapshot_rounds(cfg) -> list:
    d = league_dir(cfg)
    if not d.exists():
        return []
    return sorted(int(p.stem.split("-")[1]) for p in d.glob("snap-*.pt"))


def prune_pool(cfg, cap, anchors=2) -> list:
    rounds = snapshot_rounds(cfg)
    if len(rounds) <= cap:
        return rounds
    keep = set(rounds[:anchors]) | set(rounds[-(cap - anchors):])
    for r in rounds:
        if r not in keep:
            _snap_path(cfg, r).unlink()
    return sorted(keep)


def sample_opponent(cfg, round_n, rng):
    pool = snapshot_rounds(cfg)
    u = rng.random()
    if u < cfg.mirror_frac or not pool:
        return ("current", None)
    if u < cfg.mirror_frac + cfg.pool_frac:
        r = rng.choice(pool)
        return ("pool", str(_snap_path(cfg, r)))
    return ("random", None)


def load_opponent(path, tables, cfg) -> PolicyModel:
    p = PolicyModel(model_config_for(cfg.model_size, tables))
    sd = torch.load(path, map_location="cpu", weights_only=False)
    p.load_state_dict(sd)
    p.eval()
    return p
```

Note the mixture edge: when `not pool`, always "current" (checked first). The
`test_sample_opponent_mixture` empty-pool case asserts this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_league.py -v` → 3 PASS
Run: `python -m pytest tests/ -q` → all pass (TrainConfig field additions must
not break phase-2 tests).

- [ ] **Step 5: Commit**

```bash
git add ptcg/league.py ptcg/trainloop.py tests/test_league.py
git commit -m "league snapshot store and mixture sampler"
```

---

### Task 4: League rollout (learner-seat collection)

**Files:**
- Modify: `ptcg/rollout.py` (extend `Episode`; add `play_league_game`)
- Test: `tests/test_league_rollout.py`

**Interfaces:**
- Produces:

```python
# Episode gains two optional fields (defaults preserve phase-2 behavior):
@dataclass
class Episode:
    steps: list; result: int; rewards: tuple
    featurizer_version: int = FEATURIZER_VERSION
    decks: tuple = (None, None)          # (deck_seat0, deck_seat1) as id-lists
    collected_seats: tuple = (0, 1)      # seats whose steps are on-policy in `steps`

def play_league_game(learner, opponent, decks, tables, *, learner_seat,
                     mirror, generator=None, step_cap=5000) -> Episode
    # learner plays `learner_seat`; `opponent` (PolicyModel or "random") plays
    # the other seat. If mirror (opponent is learner), collect BOTH seats'
    # steps (collected_seats=(0,1)); else collect ONLY learner_seat's steps
    # (collected_seats=(learner_seat,)). priv_state built per step via
    # featurize_privileged with viz (as play_game does). rewards per seat from
    # result. decks recorded on the Episode.
```

- Every collected Step is featurized/encoded/sampled exactly as `play_game`
  does (same belief-tracker update ordering, same priv_viz path). The opponent
  seat still plays (so the game is legal and complete) but its steps are only
  appended when `mirror`.

- [ ] **Step 1: Write the failing test** — `tests/test_league_rollout.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.decks import deck as get_deck, all_decks
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_league_game


def test_non_mirror_collects_only_learner_seat():
    tables = build_tables()
    deck = load_sample_deck()
    learner = PolicyModel(tiny_config(tables))
    opp = PolicyModel(tiny_config(tables))     # different weights (a snapshot stand-in)
    g = torch.Generator().manual_seed(0)
    ep = play_league_game(learner, opp, (deck, list(deck)), tables,
                          learner_seat=0, mirror=False, generator=g)
    assert ep.collected_seats == (0,)
    assert all(s.player == 0 for s in ep.steps)      # only learner seat collected
    assert 5 <= len(ep.steps) <= 1000
    assert ep.result in (0, 1, 2)
    assert ep.decks[0] == deck and ep.decks[1] == list(deck)
    assert all(s.logprob <= 0 and s.logprob == s.logprob for s in ep.steps)


def test_mirror_collects_both_seats():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    ep = play_league_game(m, m, (deck, list(deck)), tables,
                          learner_seat=0, mirror=True, generator=g)
    assert ep.collected_seats == (0, 1)
    assert {s.player for s in ep.steps} <= {0, 1}
    assert any(s.player == 1 for s in ep.steps)      # both seats present


def test_learner_seat_1_and_random_opponent():
    tables = build_tables()
    deck = load_sample_deck()
    learner = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(2)
    ep = play_league_game(learner, "random", (deck, list(deck)), tables,
                          learner_seat=1, mirror=False, generator=g)
    assert ep.collected_seats == (1,) and all(s.player == 1 for s in ep.steps)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_league_rollout.py -v`
Expected: FAIL with `ImportError: cannot import name 'play_league_game'`

- [ ] **Step 3: Implement** — add the two `Episode` fields and
  `play_league_game` to `ptcg/rollout.py`. Structure (mirror `play_game`'s body,
  the trackers/priv_viz/viz_hands block is identical; the only differences are
  the per-seat actor selection and the collect condition):

```python
def play_league_game(learner, opponent, decks, tables, *, learner_seat,
                     mirror, generator=None, step_cap=5000):
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    last_obs = [s.obs, s.obs]
    seen = [False, False]
    steps = []
    try:
        while not s.done:
            if len(steps) >= step_cap:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            last_obs[me] = s.obs
            seen[me] = True
            trackers[me].update(s.obs.get("logs", []))
            actor = learner if me == learner_seat else opponent
            collect = mirror or (me == learner_seat)
            if actor == "random":
                from .engine import random_picks
                import random as _r
                # deterministic per-step rng derived from generator
                rng = _r.Random(int(torch.randint(1 << 30, (1,), generator=generator)))
                s.select(random_picks(s.obs, rng))
                continue
            ts = featurize_state(s.obs, me, decks[me], trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            if collect:
                vcur = s.viz_current()
                viz_hands = None
                if not (seen[0] and seen[1]):
                    vp = vcur.get("players") or []
                    if len(vp) == 2:
                        viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
                pv = featurize_privileged(last_obs[0], last_obs[1], decks, tables,
                                          viz=vcur, viz_hands=viz_hands)
            d = sample_select(actor, ts, es, generator)
            if collect:
                steps.append(Step(me, ts, es, d.picks, d.logprob, pv))
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    cs = (0, 1) if mirror else (learner_seat,)
    return Episode(steps, r, rewards, decks=(list(decks[0]), list(decks[1])),
                   collected_seats=cs)
```

Import `sample_select` from `.action` (already imported in rollout). Note the
random-opponent branch must NOT consume the generator differently on collect vs
non-collect steps in a way that desyncs replay — but since random-opponent steps
are never collected and the learner's own sampling always advances `generator`
identically, exactness is preserved for collected steps.

- [ ] **Step 4: Run the new test, then the suite**

Run: `python -m pytest tests/test_league_rollout.py -v` → 3 PASS
Run: `python -m pytest tests/ -q` → all pass (Episode field additions must not
break phase-2 `play_game`/serialization tests).

- [ ] **Step 5: Commit**

```bash
git add ptcg/rollout.py tests/test_league_rollout.py
git commit -m "league rollout with learner-seat collection"
```

---

### Task 5: Per-episode aux targets (multi-deck)

**Files:**
- Modify: `ptcg/ppo.py` (`aux_targets` gains per-step opponent decks)
- Test: `tests/test_ppo_multideck.py`

**Interfaces:**
- Produces (backward-compatible overload):

```python
def aux_targets(steps, tables, opp_decks) -> (pd, dl, hd)
    # opp_decks: EITHER a single list[int] (phase-2 behavior, one deck for all
    # steps) OR a list of per-step list[int] (len == len(steps)). The decklist
    # target dl[i] is the multiset of the opponent's deck for step i.
```

- Detection: if `opp_decks` and `isinstance(opp_decks[0], int)` → treat as one
  shared deck (phase-2 path). Else treat as per-step lists.

- [ ] **Step 1: Write the failing test** — `tests/test_ppo_multideck.py`:

```python
import torch
from ptcg.cards import build_tables, card_row
from ptcg.ppo import aux_targets
from ptcg.rollout import play_league_game
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config


class _S:  # minimal Step stand-in with the fields aux_targets reads
    def __init__(self, player, state, priv):
        self.player, self.state, self.priv_state = player, state, priv


def test_aux_targets_per_step_decks():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    steps = ep.steps[:6]
    deckA = deck
    deckB = list(reversed(deck))                 # a different "opponent" deck id-list
    per = [deckB if s.player == 0 else deckA for s in steps]
    pd, dl, hd = aux_targets(steps, tables, per)
    assert dl.shape == (len(steps), tables.n_rows)
    # a step whose opponent deck is deckB has that deck's row counts
    i = next(k for k, s in enumerate(steps) if s.player == 0)
    from collections import Counter
    exp = Counter(card_row(c, tables.n_rows) for c in deckB)
    assert dl[i, next(iter(exp))] > 0


def test_aux_targets_single_deck_backcompat():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    pd, dl, hd = aux_targets(ep.steps[:4], tables, deck)   # single list[int]
    assert dl.shape == (4, tables.n_rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ppo_multideck.py -v`
Expected: FAIL (per-step decks currently unsupported — shape/typing error).

- [ ] **Step 3: Implement** — change `aux_targets` so the decklist target is
  built per step from the step's opponent deck. Keep prize-diff and hand targets
  exactly as they are (hand comes from the privileged state). Precise change:

```python
def aux_targets(steps, tables, opp_decks):
    n_rows = tables.n_rows
    B = len(steps)
    pd = torch.zeros(B); dl = torch.zeros(B, n_rows); hd = torch.zeros(B, n_rows)
    shared = bool(opp_decks) and isinstance(opp_decks[0], int)
    shared_vec = None
    if shared:
        shared_vec = torch.zeros(n_rows)
        for cid in opp_decks:
            shared_vec[card_row(cid, n_rows)] += 1.0
    for i, s in enumerate(steps):
        num = s.state.numeric
        pd[i] = float(num[_PSUM_OPP, F_PRIZEN] - num[_PSUM_SELF, F_PRIZEN]) * 6.0
        if shared:
            dl[i] = shared_vec
        else:
            for cid in opp_decks[i]:
                dl[i, card_row(cid, n_rows)] += 1.0
        opp_owner = OWNER_OPP if s.player == 0 else OWNER_SELF
        pv = s.priv_state
        rows = np.where((pv.zone[:pv.n] == AREA_HAND)
                        & (pv.owner[:pv.n] == opp_owner)
                        & (pv.kind[:pv.n] == KIND_ENTITY))[0]
        for r in rows:
            hd[i, int(pv.card[r])] += 1.0
    return pd, dl, hd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ppo_multideck.py -v` → 2 PASS
Run: `python -m pytest tests/test_ppo_units.py tests/test_learner.py -v` →
still pass (single-deck path unchanged).

- [ ] **Step 5: Commit**

```bash
git add ptcg/ppo.py tests/test_ppo_multideck.py
git commit -m "per-episode aux decklist targets"
```

---

### Task 6: League round worker + learner integration

**Files:**
- Modify: `ptcg/actors.py` (add `league_round_worker`), `ptcg/trainloop.py`
  (`learner_update` builds per-step opp_decks from episodes)
- Test: `tests/test_league_worker.py`

**Interfaces:**
- Produces:

```python
# ptcg/actors.py
def league_round_worker(args) -> dict
    # args = (cfg_json, round_n, actor_idx, n_games, ckpt_path)
    # loads current policy from ckpt_path; for each game: rng from game_seed;
    # sample_opponent(cfg, round_n, rng) -> (kind, path); deck per side sampled
    # uniformly from train_decks(); learner_seat = rng-chosen; mirror iff
    # kind=="current". Opponent: current policy (mirror) / load_opponent(path) /
    # "random". play_league_game(...); save each Episode to round_dir. Returns
    # {"games","steps","results","wall_s"} like collect_round_worker.
```

- `learner_update(policy, critic, optim, episodes, cfg, tables, opp_deck)`:
  change so it derives per-step opponent decks from each Episode's `.decks` and
  the step's `.player` (opp deck = `ep.decks[1 - step.player]`), and passes that
  per-step list to `aux_targets`. When an Episode has no `.decks` (phase-2
  Episode), fall back to the passed `opp_deck` (single list) — backward compat.
  Nothing else in `learner_update` changes; assemble_advantages already handles
  episodes whose steps are a single seat (opponent seat simply has no rows).

- [ ] **Step 1: Write the failing test** — `tests/test_league_worker.py`:

```python
import json
from dataclasses import asdict
import torch
from ptcg.actors import league_round_worker
from ptcg.cards import build_tables
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.league import snapshot
from ptcg.trainloop import (TrainConfig, save_checkpoint, load_round,
                            learner_update)


def _seed_run(tmp_path, games, **kw):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                      games_per_round=games, actors=1, **kw)
    p = PolicyModel(tiny_config(tables)); c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    return tables, cfg, p


def test_league_worker_writes_mixed_games(tmp_path):
    # pool has one snapshot so non-mirror pool games occur
    tables, cfg, p = _seed_run(tmp_path, 3, mirror_frac=0.34, pool_frac=0.66,
                               random_frac=0.0, pool_cap=18)
    snapshot(cfg, 0, p)
    stats = league_round_worker(
        (json.dumps(asdict(cfg)), 1, 0, 3, str(tmp_path / "checkpoint-0000.pt")))
    assert stats["games"] == 3 and stats["steps"] > 10
    eps = load_round(cfg, 1)
    assert len(eps) == 3
    # every episode records its decks and collected seats
    for e in eps:
        assert e.decks[0] and e.decks[1]
        assert set(e.collected_seats) <= {0, 1}
        assert all(s.player in e.collected_seats for s in e.steps)


def test_learner_update_consumes_multideck_episodes(tmp_path):
    tables, cfg, p = _seed_run(tmp_path, 2, mirror_frac=1.0, pool_frac=0.0,
                               random_frac=0.0, epochs=1, minibatch=64)
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=3e-4)
    league_round_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                         str(tmp_path / "checkpoint-0000.pt")))
    eps = load_round(cfg, 0)
    m = learner_update(p, c, opt, eps, cfg, tables, opp_deck=None)
    assert m["ratio_drift"] < 1e-6 and m["steps"] > 10
    for k in ("loss_pg", "loss_aux", "entropy"):
        assert torch.isfinite(torch.tensor(m[k])), k
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_league_worker.py -v`
Expected: FAIL with `ImportError: cannot import name 'league_round_worker'`

- [ ] **Step 3: Implement** `league_round_worker` in `ptcg/actors.py` and the
  per-step-opp-deck change in `learner_update`.

`league_round_worker` (top-level, spawn-safe; mirror the structure of
`collect_round_worker`):

```python
def league_round_worker(args):
    import random as _r
    from .decks import deck as get_deck, train_decks
    from .league import sample_opponent, load_opponent
    from .rollout import play_league_game
    from .trainloop import save_game
    cfg_json, round_n, actor_idx, n_games, ckpt_path = args
    cfg = TrainConfig(**json.loads(cfg_json))
    tables = build_tables()
    policy = PolicyModel(model_config_for(cfg.model_size, tables))
    load_checkpoint(ckpt_path, policy, _NullCritic(), optim=None)
    policy.eval()
    rd = round_dir(cfg, round_n); rd.mkdir(parents=True, exist_ok=True)
    names = train_decks()
    t0 = time.perf_counter(); steps = 0; results = []
    for g in range(n_games):
        seed = game_seed(cfg, round_n, actor_idx, g)
        gen = torch.Generator().manual_seed(seed)
        rng = _r.Random(seed)
        kind, path = sample_opponent(cfg, round_n, rng)
        da = get_deck(rng.choice(names)); db = get_deck(rng.choice(names))
        learner_seat = rng.randint(0, 1)
        mirror = kind == "current"
        if kind == "pool":
            opp = load_opponent(path, tables, cfg)
        elif kind == "random":
            opp = "random"
        else:
            opp = policy
        opponent = opp if kind != "pool" else opp
        with torch.no_grad():
            ep = play_league_game(policy, opponent, (da, db), tables,
                                  learner_seat=learner_seat, mirror=mirror,
                                  generator=gen, step_cap=cfg.step_cap)
        save_game(rd / f"a{actor_idx}-g{g}.pt", ep)
        steps += len(ep.steps); results.append(ep.result)
    return {"games": n_games, "steps": steps, "results": results,
            "wall_s": time.perf_counter() - t0}
```

(Clean up the redundant `opp`/`opponent` lines when implementing — the point is:
mirror→`policy`, pool→`load_opponent`, random→`"random"`.)

`learner_update` change — replace the single `aux_targets(steps, tables,
opp_deck)` call with per-step decks derived from episodes:

```python
    # build per-step opponent decks from each episode's recorded decks
    opp_decks = []
    for ep in episodes:
        ep_decks = getattr(ep, "decks", (None, None))
        for s in ep.steps:
            od = ep_decks[1 - s.player] if ep_decks[1 - s.player] is not None else opp_deck
            opp_decks.append(od)
    # NOTE: assemble_advantages returns `steps` in the SAME episode-then-step
    # order this loop uses, so opp_decks aligns with `steps`. Verify by asserting
    # len(opp_decks) == len(steps) before calling aux_targets.
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_decks)
```

Confirm the ordering: `assemble_advantages` builds `steps = [s for ep in
episodes for s in ep.steps]` — identical flatten order to the `opp_decks` loop.
Add `assert len(opp_decks) == len(steps)` guarding the alignment.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_league_worker.py -v` → 2 PASS (minutes)
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add ptcg/actors.py ptcg/trainloop.py tests/test_league_worker.py
git commit -m "league round worker and multideck learner update"
```

---

### Task 7: Trainloop wiring — league snapshotting + orchestration

**Files:**
- Modify: `ptcg/trainloop.py` (`train()` uses `league_round_worker`; snapshots
  each `snapshot_every`; prunes pool)
- Test: `tests/test_league_train_integration.py`

**Interfaces:**
- `train(cfg, max_rounds)` changes: (1) collection uses `league_round_worker`
  via `run_actor_pool(..., worker=league_round_worker)`; (2) after
  `save_checkpoint(rnd+1)`, when `(rnd+1) % cfg.snapshot_every == 0`,
  `snapshot(cfg, rnd+1, policy)` then `prune_pool(cfg, cfg.pool_cap)`; (3) resume
  restores the league dir as-is (snapshots are on disk; nothing to rebuild).
  Everything else (metrics, resume, eval hook, checkpoint pruning) unchanged.

- [ ] **Step 1: Write the failing test** — `tests/test_league_train_integration.py`:

```python
from ptcg.trainloop import TrainConfig, latest_checkpoint, read_metrics, train
from ptcg.league import snapshot_rounds


def _cfg(tmp_path):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                       games_per_round=2, actors=1, epochs=1, minibatch=64,
                       eval_every=999, snapshot_every=1, pool_cap=4,
                       mirror_frac=0.5, pool_frac=0.5, random_frac=0.0,
                       device="cpu", seed=5)


def test_league_train_snapshots_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg, max_rounds=2)
    assert latest_checkpoint(cfg)[0] == 2
    # snapshot_every=1 -> pool has snapshots for rounds 1 and 2
    assert snapshot_rounds(cfg) == [1, 2]
    rows = [r for r in read_metrics(cfg) if r["kind"] == "train"]
    assert [int(r["round"]) for r in rows] == [0, 1]
    train(cfg, max_rounds=3)               # resume: no redo, pool grows
    assert latest_checkpoint(cfg)[0] == 3
    assert 3 in snapshot_rounds(cfg)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_league_train_integration.py -v`
Expected: FAIL (train still uses `collect_round_worker`; no snapshots created).

- [ ] **Step 3: Implement** — in `train()`, change the collection call and add
  snapshotting. Concretely:

```python
    # in the per-round loop, replace the collection line:
    stats = run_actor_pool(cfg, rnd, ck, worker=league_round_worker)
    # ... learner_update, save_checkpoint(rnd+1), append_metrics unchanged ...
    # after save_checkpoint(cfg, rnd + 1, ...):
    if (rnd + 1) % cfg.snapshot_every == 0:
        from .league import snapshot, prune_pool
        snapshot(cfg, rnd + 1, policy)
        prune_pool(cfg, cfg.pool_cap)
```

Add `from .actors import league_round_worker` to the imports inside `train()`
(alongside the existing `run_actor_pool` import — keep them function-local to
avoid the actors→trainloop circular import). `learner_update` is called with
`opp_deck=None` now (episodes carry their own decks); confirm the signature call
site passes `opp_deck=None` (or the sample deck — either works since episodes
override it).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_league_train_integration.py -v` → 1 PASS
(a few minutes — real tiny games with a pool snapshot)
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add ptcg/trainloop.py tests/test_league_train_integration.py
git commit -m "league training loop with snapshotting"
```

---

### Task 8: Evaluation — per-deck + vs-SD-champ

**Files:**
- Modify: `ptcg/actors.py` (`league_eval_worker`), `ptcg/trainloop.py`
  (`_eval_due` adds per-deck-vs-random and vs-SD-champ across the portfolio;
  `METRIC_FIELDS` gains columns)
- Test: `tests/test_league_eval.py`

**Interfaces:**
- Produces:

```python
# ptcg/actors.py
def league_eval_worker(args) -> dict
    # args = (cfg_json, round_n, actor_idx, n_games, ckpt_path, opp_spec, deck_name)
    # like eval_worker but BOTH sides play PORTFOLIO[deck_name]; opp_spec is
    # "random" | "<sd_champ ckpt path>" | "<pool snap path>". Returns {"wins","games"}.
# ptcg/trainloop.py — _eval_due additionally emits:
#   wr_random_mean (avg per-deck vs random over portfolio),
#   wr_champ_nonsample (avg vs SD-champ over non-sample decks),
#   wr_champ_sample (vs SD-champ on the sample deck).
# METRIC_FIELDS extended with: wr_random_mean, wr_champ_nonsample, wr_champ_sample
```

- The frozen SD-champ path is `cfg.sd_champ_ckpt` (a checkpoint file). If empty
  or missing, the champ metrics are left blank (eval still runs vs random).

- [ ] **Step 1: Write the failing test** — `tests/test_league_eval.py`:

```python
import json
from dataclasses import asdict
import torch
from ptcg.actors import league_eval_worker
from ptcg.cards import build_tables
from ptcg.decks import SAMPLE
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, save_checkpoint


def test_league_eval_worker_per_deck(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    p = PolicyModel(tiny_config(tables)); c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()))
    save_checkpoint(cfg, 0, p, c, opt)
    out = league_eval_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                              str(tmp_path / "checkpoint-0000.pt"),
                              "random", SAMPLE))
    assert out["games"] == 2 and 0 <= out["wins"] <= 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_league_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'league_eval_worker'`

- [ ] **Step 3: Implement** `league_eval_worker` (both sides play the named
  deck; opponent = random or a loaded checkpoint policy; alternate seats by
  parity), and extend `_eval_due` to, when due: for each portfolio deck play
  `cfg.eval_games_random // len(portfolio)` games vs random (aggregate →
  `wr_random_mean`); if `cfg.sd_champ_ckpt` exists, for each deck play
  `cfg.eval_games_ckpt // len(portfolio)` vs the champ, aggregating the sample
  deck separately (`wr_champ_sample`) from the mean over the rest
  (`wr_champ_nonsample`). Keep the phase-2 `wr_random`/`wr_ck5`/`wr_ck15` columns
  as-is (they still run on the sample deck) so existing plots keep working. Add
  the three new names to `METRIC_FIELDS`.

Reuse `play_versus` for the games (it already alternates via `model_seat`, both
sides same deck). `league_eval_worker` differs from `eval_worker` only by taking
`deck_name` and using `PORTFOLIO[deck_name]` for both `decks`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_league_eval.py -v` → 1 PASS
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add ptcg/actors.py ptcg/trainloop.py tests/test_league_eval.py
git commit -m "per-deck and vs-champ evaluation"
```

---

### Task 9: CLI, plots, SD-champ freeze, smoke runs

**Files:**
- Modify: `scripts/train.py` (flags auto-extend from TrainConfig — already
  generic; verify the new fields appear), `scripts/plot_run.py` (add
  `wr_champ_nonsample`/`wr_champ_sample`/`wr_random_mean` curves)
- Create: `scripts/freeze_champ.py` (copy the phase-2 champion policy to a
  stable path), `benchmarks/RESULTS-league-smoke.md`

- [ ] **Step 1: Freeze the SD-champ** — `scripts/freeze_champ.py`:

```python
"""Freeze the phase-2 single-deck champion as the fixed generalization baseline.
Usage: venv-train\\Scripts\\python scripts\\freeze_champ.py runs/phase2-a/checkpoint-0031.pt champ/sd-champ.pt"""
import shutil, sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
Path(dst).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
print("froze", src, "->", dst)
```

Run it: `venv-train\Scripts\python scripts\freeze_champ.py runs\phase2-a\checkpoint-0031.pt champ\sd-champ.pt`
(`champ/` is git-ignored like `runs/`; add it to `.gitignore`.) The training
launch passes `--sd-champ-ckpt champ/sd-champ.pt`.

- [ ] **Step 2: Update `scripts/plot_run.py`** — add an eval subplot plotting
  `wr_champ_nonsample` (the headline), `wr_champ_sample`, and `wr_random_mean`
  vs round, with a 0.60 reference line for the generalization criterion. Verify
  `scripts/train.py --help` lists the new flags (`--mirror-frac`, `--pool-frac`,
  `--random-frac`, `--snapshot-every`, `--pool-cap`, `--sd-champ-ckpt`).

- [ ] **Step 3: CPU smoke run** (base interpreter, real spawn):

Run: `python scripts/train.py --run-id league-smoke --model-size tiny --games-per-round 8 --actors 2 --max-rounds 3 --minibatch 64 --epochs 1 --snapshot-every 1 --eval-every 2 --eval-games-random 8 --eval-games-ckpt 4 --mirror-frac 0.34 --pool-frac 0.66 --random-frac 0.0 --device cpu`
Expected: completes; `runs/league-smoke/league/` has snapshots; metrics has
train rows + an eval row with `wr_random_mean` populated; no `rounds/` residue.
Then `python scripts/plot_run.py runs/league-smoke` → plots render.

- [ ] **Step 4: GPU smoke run** (training venv, real config shape):

Run: `venv-train\Scripts\python scripts\train.py --run-id league-gpu --model-size student --games-per-round 12 --actors 3 --max-rounds 2 --minibatch 128 --snapshot-every 1 --eval-every 999 --sd-champ-ckpt champ\sd-champ.pt --device cuda`
Expected: completes; ratio_drift < 1e-3; league snapshots written; finite
losses. Record `ratio_drift` and per-round wall time.

- [ ] **Step 5: Record + commit**

Write `benchmarks/RESULTS-league-smoke.md` (both commands, wall times,
games/steps, GPU ratio_drift, snapshot/pool behavior). Delete `runs/league-*`.
Add `champ/` to `.gitignore`.

```bash
git add scripts/train.py scripts/plot_run.py scripts/freeze_champ.py .gitignore benchmarks/RESULTS-league-smoke.md
git commit -m "league cli, champ freeze, plots and smoke runs"
```

---

### Task 10: Qualification-run playbook

**Files:**
- Create: `docs/phase3-run.md`

- [ ] **Step 1: Write `docs/phase3-run.md`** containing exactly:

```markdown
# Phase-3 qualification run

Prereqs: freeze the SD-champ once —
`venv-train\Scripts\python scripts\freeze_champ.py runs\phase2-a\checkpoint-0031.pt champ\sd-champ.pt`
(if `runs/phase2-a` was deleted, resume phase-2 briefly to regenerate a
checkpoint, or point at any saved phase-2 checkpoint).

Launch FULLY DETACHED (never as a child of a transient shell/agent — a phase-2
lesson), from the repo root:

    venv-train\Scripts\python scripts\train.py --run-id phase3-a --device cuda --seed 1 --minibatch 128 --sd-champ-ckpt champ\sd-champ.pt --games-per-round 256

Resume after any interruption: the same command (loads the latest checkpoint and
the on-disk league pool).

Monitor: `python scripts/plot_run.py runs/phase3-a`

Success criteria (spec §Success criteria — all four, sized-sample stats):
1. Generalization: wr_champ_nonsample > 0.60 on two consecutive evals.
2. Sustained gradient: current beats the 15-rounds-back league snapshot > 0.55
   for materially longer than the single-deck mirror did (which stalled ~round
   10-15).
3. Per-deck competence: >= 0.70 vs random on every trained deck by the end.
4. Stability: zero ratio-gate aborts, zero NaN, a clean kill+resume with
   contiguous metrics AND an intact league pool.

Disk: student checkpoints ~103 MB (pruned to {0, mult of 5, last 2}); league
snapshots are policy-only (~52 MB) capped at pool_cap (~18) with anchor
retention — budget ~10-15 GB. `runs/` and `champ/` are git-ignored.

On success: record run id, final eval row, and wall-clock in
benchmarks/RESULTS-league-smoke.md under a "qualification" heading. On failure:
capture metrics.csv + plots and open a debugging session — do not reinterpret
the criteria.
```

- [ ] **Step 2: Launch** (only if the machine is free and the SD-champ is
  frozen; otherwise hand the command to the operator). Verify the first train
  row and one kill+resume cycle (criterion 4), leave running.

- [ ] **Step 3: Commit**

```bash
git add docs/phase3-run.md
git commit -m "phase 3 qualification run playbook"
```

---

## Self-review notes (kept for the record)

- Spec coverage: deck portfolio (real-meta primary, engine-validated, ≥6
  distinct) → Tasks 1-2; league snapshot store + mixture sampling + retention →
  Task 3; learner-seat-only collection (the correctness rule) → Task 4;
  multi-deck aux targets → Task 5; league round worker + learner integration →
  Task 6; snapshotting/orchestration/resume → Task 7; per-deck + vs-SD-champ eval
  and success-criteria machinery → Task 8; CLI/champ-freeze/plots/smoke → Task 9;
  qualification playbook + criteria → Task 10. Reused phase-2 verbatim: batched
  replay, PPO losses, GAE, ratio gate, checkpoint IO/resume, prune_checkpoints.
- Correctness rule enforced structurally: `play_league_game` only appends
  learner-seat steps for non-mirror games (Task 4 tests assert
  `all(s.player == learner_seat)`); the ratio contract holds because opponent
  steps never enter `assemble_advantages`/replay.
- TrainConfig additions (`mirror_frac`, `pool_frac`, `random_frac`,
  `snapshot_every`, `pool_cap`, `sd_champ_ckpt`) land in Task 3 (needed by the
  Task-3 tests) and are wired into `train()` in Task 7; `scripts/train.py`
  generates their flags automatically (all float/int/str — no bool footgun).
- Deferred to phase 4 (spec §out of scope): distillation, IS-MCTS search, clock,
  submission packaging, deck selection, prioritized opponent selection, held-out
  decks.
- Known risk called out in-task: deck sourcing depends on web/user data (Task 2)
  — the resolver + validation + fallback assembly make it mechanical, and the
  count target (≥6) is the acceptance gate.
- Scoping call (record as a deviation if you disagree): the spec's cross-deck
  matchup *matrix* is a periodic diagnostic that only "seeds later deck
  selection" (a phase-4 concern), so it is NOT baked into the per-round eval
  (which would multiply eval cost by |portfolio|²). It is producible on demand
  by a throwaway script reusing `play_versus` + `PORTFOLIO` (the pattern of the
  phase-2 `diag_plateau.py`). The per-round eval covers the success-criteria
  metrics (per-deck-vs-random, vs-SD-champ); the matrix is run manually when
  deck selection is needed.
```
