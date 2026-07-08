# Competition submission (phase-3 generalist) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the trained phase-3 generalist policy into a Kaggle-ready
`agent(obs_dict)` submission that returns a fixed deck at deck-selection and plays
legal moves via the policy, per
`docs/superpowers/specs/2026-07-08-competition-submission-design.md`.

**Architecture:** Author the agent in a committed `submission_src/`; a build script
assembles a self-contained, git-ignored `dist/submission/` bundle (agent + trimmed
`ptcg` inference package + sample `cg/` + policy weights + deck.csv). Inference reuses
the exact training/eval path (featurize → encode_select → sample_select), wrapped so it
always returns a legal pick.

**Tech Stack:** existing `ptcg` package; PyTorch (CPU inference); the competition `cg`
engine module; Python (base interpreter for tests/scripts). Card metadata comes from the
native `cg` module, not the CSV.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-competition-submission-design.md`. On conflict
  the spec wins; deviations go in the final report.
- Submission contract (fixed): `agent(obs_dict: dict) -> list[int]`; return 60 card IDs
  when `obs_dict.get("select") is None`, else option indices each in
  `[0, len(select.option))`, count in `[select.minCount, select.maxCount]`, no duplicates.
- Inference path is byte-identical to training/eval: `featurize_state(obs, me, own_deck,
  tracker.snapshot(), tables)` → `encode_select(obs, ts, tables)` →
  `sample_select(model, ts, es, generator)` → `.picks`. `me = obs["current"]["yourIndex"]`.
  Move choice is STOCHASTIC sampling (matches measured win-rates).
- The agent MUST never raise: wrap inference in `try/except` and fall back to a legal
  random pick (`random.sample(range(len(option)), maxCount)`).
- Reset per-game state (belief tracker, seat) whenever `select is None` (a new game).
- Model is student size: `PolicyModel(student_config(tables))`, weights = the `"policy"`
  state_dict from `runs/phase3-a/checkpoint-0120.pt` (frozen at `champ/phase3-generalist-r120.pt`).
- Never modify anything under `pokemon-tcg-ai-battle/`; `cg/` is COPIED, not edited.
- One engine battle per process: run the smoke/round-robin scripts one at a time, never
  in parallel, never alongside pytest.
- Commit style: short lowercase messages, directly on `main`, no co-author lines. Base
  interpreter `python` for everything here (CPU). Big artifacts go under git-ignored paths
  (`dist/`, `submission_src/policy.pt`).

## Existing interfaces this plan builds on (do not change)

```python
# ptcg/cards.py:     build_tables() -> CardTables            # loads native cg, reads card metadata
# ptcg/model.py:     student_config(tables); class PolicyModel(nn.Module)   # .load_state_dict(sd)
# ptcg/featurize.py: featurize_state(obs, me, own_deck, belief, tables) -> TokenizedState
#                    encode_select(obs: dict, ts, tables) -> EncodedSelect
# ptcg/action.py:    sample_select(model, ts, es, generator=None) -> SelectDecision  # .picks, .logprob
# ptcg/tracker.py:   class BeliefTracker(my_index); .update(logs: list); .snapshot() -> BeliefSnapshot
# ptcg/engine.py:    engine_dir() honors $PTCG_ENGINE_DIR; BattleSession(d0,d1) (.obs/.done/.result/.select/.close)
#                    load_sample_deck() -> list[int]; random_picks(obs, rng)
# ptcg/actors.py:    play_versus(model, opponent, tables, decks, generator, model_seat, step_cap=5000) -> int
# ptcg/decks.py:     PORTFOLIO: dict[str,list[int]]; all_decks(); deck(name)->list[int]; SAMPLE
```

## File Structure

```
submission_src/main.py     — the agent(obs_dict) entrypoint (source of truth)   [committed]
submission_src/deck.csv    — chosen deck, 60 IDs one per line                   [committed]
submission_src/README.md   — one-paragraph usage note                            [committed]
submission_src/policy.pt   — extracted policy weights (~52 MB)         [git-ignored]
scripts/extract_policy.py  — checkpoint "policy" state_dict -> submission_src/policy.pt
scripts/test_submission.py — end-to-end smoke: agent drives a real BattleSession seat
scripts/deck_roundrobin.py — generalist deck-vs-deck over the portfolio -> best deck
scripts/make_submission.py — assemble dist/submission/ + zip; isolated self-containment check
tests/test_submission_agent.py — fast pytest: deck-return + fallback branches (no engine/weights)
dist/submission/, dist/submission.zip                                   [git-ignored]
```

---

### Task 1: Policy extraction + the agent module + fast unit tests

**Files:**
- Create: `scripts/extract_policy.py`, `submission_src/main.py`, `submission_src/deck.csv`
- Create: `tests/test_submission_agent.py`
- Modify: `.gitignore` (add `submission_src/policy.pt`)

**Interfaces:**
- Produces: `submission_src/main.py` exposing `agent(obs_dict: dict) -> list[int]` and
  module globals it manages internally.

- [ ] **Step 1: Write `scripts/extract_policy.py`**

```python
"""Extract the policy-only state_dict from a training checkpoint.
Usage: python scripts/extract_policy.py runs/phase3-a/checkpoint-0120.pt submission_src/policy.pt"""
import os
import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["policy"] if isinstance(ck, dict) and "policy" in ck else ck
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save({k: v.cpu() for k, v in sd.items()}, dst)
    print(f"wrote {dst} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/extract_policy.py runs/phase3-a/checkpoint-0120.pt submission_src/policy.pt`
Expected: prints `wrote submission_src/policy.pt (N tensors)`; file ~52 MB.
Then add `submission_src/policy.pt` to `.gitignore`.

- [ ] **Step 3: Write a placeholder `submission_src/deck.csv`**

Generate it from the sample deck for now (finalized in Task 3):
`python -c "from ptcg.engine import load_sample_deck; open('submission_src/deck.csv','w').write('\n'.join(map(str, load_sample_deck())))"`
Expected: 60 lines, one integer per line.

- [ ] **Step 4: Write `submission_src/main.py`**

```python
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
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
        sel = obs_dict["select"]
        n = len(sel["option"])
        legal = (picks and len(set(picks)) == len(picks)
                 and all(0 <= p < n for p in picks)
                 and sel["minCount"] <= len(picks) <= sel["maxCount"])
        return picks if legal else _fallback(obs_dict)
    except Exception:
        return _fallback(obs_dict)
```

- [ ] **Step 5: Write the failing test** — `tests/test_submission_agent.py`:

```python
import importlib.util
import os

import pytest

_MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "submission_src", "main.py")


def _load_agent():
    spec = importlib.util.spec_from_file_location("submission_main", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deck_selection_returns_60_ids():
    mod = _load_agent()
    deck = mod.agent({"select": None, "current": None, "logs": []})
    assert isinstance(deck, list) and len(deck) == 60
    assert all(isinstance(c, int) for c in deck)


def test_illegal_or_broken_obs_falls_back_to_legal_pick():
    mod = _load_agent()
    # An obs whose featurization will fail (missing 'current') must still yield a
    # legal selection from the option list, never an exception.
    obs = {"select": {"option": list(range(5)), "minCount": 1, "maxCount": 2},
           "logs": []}
    picks = mod.agent(obs)
    assert isinstance(picks, list)
    assert 1 <= len(picks) <= 2
    assert len(set(picks)) == len(picks)
    assert all(0 <= p < 5 for p in picks)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_submission_agent.py -v`
Expected: 2 PASS. (`test_deck_selection...` needs no model; the fallback test forces the
`except` path because the obs lacks `current` — proving the agent never raises.)
Then run the full suite once: `python -m pytest tests/ -q` → all pass (no regressions;
this task adds files only).

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_policy.py submission_src/main.py submission_src/deck.csv tests/test_submission_agent.py .gitignore
git commit -m "submission agent module and policy extraction"
```

---

### Task 2: End-to-end smoke test (agent drives a real game)

**Files:**
- Create: `scripts/test_submission.py`

**Interfaces:**
- Consumes: `submission_src/main.py:agent`, `ptcg.engine.BattleSession/random_picks`,
  `ptcg.engine.load_sample_deck`.

- [ ] **Step 1: Write `scripts/test_submission.py`**

```python
"""End-to-end smoke: the submission agent drives one seat of real BattleSession games.
Proves obs-schema compatibility + legality + latency. Run from repo root (base python),
one engine process at a time. Usage: python scripts/test_submission.py [n_games]"""
import importlib.util
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ptcg.engine import BattleSession, load_sample_deck, random_picks


def load_agent():
    path = os.path.join(REPO, "submission_src", "main.py")
    spec = importlib.util.spec_from_file_location("submission_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_legal(picks, sel):
    n = len(sel["option"])
    return (isinstance(picks, list) and picks
            and len(set(picks)) == len(picks)
            and all(isinstance(p, int) and 0 <= p < n for p in picks)
            and sel["minCount"] <= len(picks) <= sel["maxCount"])


def play(agent, my_seat, my_deck, opp_deck, seed):
    rng = random.Random(seed)
    s = BattleSession(my_deck if my_seat == 0 else opp_deck,
                      opp_deck if my_seat == 0 else my_deck)
    lat = []
    try:
        while not s.done:
            me = s.select_player
            if me == my_seat:
                t0 = time.perf_counter()
                picks = agent(s.obs)
                lat.append(time.perf_counter() - t0)
                assert is_legal(picks, s.obs["select"]), (picks, s.obs["select"])
                s.select(picks)
            else:
                s.select(random_picks(s.obs, rng))
        return s.result, lat
    finally:
        s.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    mod = load_agent()
    # The agent's OWN deck is what it declares at deck selection; the engine deals it
    # that deck, so the smoke must play my_deck = the declared deck (NOT the sample deck),
    # or featurization would mismatch the dealt cards once deck.csv changes in Task 3.
    my_deck = mod.agent({"select": None, "current": None, "logs": []})
    assert len(my_deck) == 60
    opp_deck = load_sample_deck()   # a fixed opponent deck for the smoke
    wins, done, all_lat = 0, 0, []
    for g in range(n):
        my_seat = g % 2
        mod.agent({"select": None, "current": None, "logs": []})  # reset per-game state
        result, lat = play(mod.agent, my_seat, my_deck, opp_deck, seed=1000 + g)
        done += 1
        if result == my_seat:
            wins += 1
        all_lat += lat
    all_lat.sort()
    p50 = all_lat[len(all_lat) // 2]
    p95 = all_lat[int(len(all_lat) * 0.95)]
    print(f"games={done} wins={wins} winrate={wins/done:.3f} "
          f"moves={len(all_lat)} latency p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms")
    print("OK — all agent picks legal, all games completed" if done == n else "INCOMPLETE")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke** (needs `submission_src/policy.pt` from Task 1)

Run: `python scripts/test_submission.py 6`
Expected: completes; every move legal (no AssertionError); prints a win rate vs random
that is comfortably > 0.5 and p50/p95 per-move latency. Record the output. If a move
raises or is illegal, STOP (obs-schema mismatch) and report.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_submission.py
git commit -m "submission end-to-end smoke test"
```

---

### Task 3: Deck round-robin → pick and set the submission deck

**Files:**
- Create: `scripts/deck_roundrobin.py`
- Modify: `submission_src/deck.csv` (overwrite with the chosen deck)

**Interfaces:**
- Consumes: `ptcg.actors.play_versus`, `ptcg.cards.build_tables`, `ptcg.model`
  (`PolicyModel`, `student_config`), `ptcg.decks` (`all_decks`, `deck`).

- [ ] **Step 1: Write `scripts/deck_roundrobin.py`**

```python
"""Pick the submission deck: play the generalist deck-vs-deck across the portfolio and
report each deck's average win rate across the field. Run from repo root (base python),
one engine process at a time. Usage: python scripts/deck_roundrobin.py [games_per_pair]"""
import os
import sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ptcg.actors import play_versus
from ptcg.cards import build_tables
from ptcg.decks import all_decks, deck as get_deck
from ptcg.model import PolicyModel, student_config

CKPT = os.path.join(REPO, "submission_src", "policy.pt")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tables = build_tables()
    model = PolicyModel(student_config(tables))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    names = all_decks()
    wins = {a: 0 for a in names}
    games = {a: 0 for a in names}
    seed = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = get_deck(a), get_deck(b)
            for g in range(n):
                seed += 1
                gen = torch.Generator().manual_seed(seed)
                # deck a on seat g%2, deck b on the other; play_versus returns 1 iff seat0-model won
                seat = g % 2
                decks = (da, db) if seat == 0 else (db, da)
                with torch.no_grad():
                    r = play_versus(model, model, tables, decks, gen, model_seat=seat)
                # r==1 means the seat-`seat` player (deck a) won
                wins[a] += r
                wins[b] += (1 - r)
                games[a] += 1
                games[b] += 1
    rows = sorted(((wins[a] / games[a], a) for a in names), reverse=True)
    print(f"{'deck':<26}{'winrate':>9}{'games':>8}")
    for wr, a in rows:
        print(f"{a:<26}{wr:>9.3f}{games[a]:>8}")
    print(f"\nBEST: {rows[0][1]}  ({rows[0][0]:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and pick the deck**

Run: `python scripts/deck_roundrobin.py 12`
Expected (~10-25 min CPU): a table of per-deck win rates and a `BEST:` line. Record the
table. (Note in your report: `play_versus` treats a draw as a loss for `model_seat`; that
is acceptable for ranking. If two decks are within noise, prefer the more consistent/in-era
archetype.)

- [ ] **Step 3: Write the chosen deck to `submission_src/deck.csv`**

Replace `submission_src/deck.csv` with the winning deck's 60 IDs:
`python -c "from ptcg.decks import deck; open('submission_src/deck.csv','w').write('\n'.join(map(str, deck('<BEST-name>'))))"`
Expected: 60 lines. Confirm with `python -m pytest tests/test_submission_agent.py -v`
(the deck-return test still passes with the new deck).

- [ ] **Step 4: Commit**

```bash
git add scripts/deck_roundrobin.py submission_src/deck.csv
git commit -m "deck round-robin and chosen submission deck"
```

---

### Task 4: Bundle assembly + self-containment check + finalize

**Files:**
- Create: `scripts/make_submission.py`, `submission_src/README.md`

**Interfaces:**
- Produces: `dist/submission/` (self-contained) and `dist/submission.zip`.

- [ ] **Step 1: Write `submission_src/README.md`**

```markdown
# Phase-3 generalist submission

`main.py` implements `agent(obs_dict)`: returns the 60-card `deck.csv` at deck selection,
then plays via the trained generalist policy (`policy.pt`, from phase3-a checkpoint-0120)
using the `ptcg` inference package and the competition `cg` module. Assembled by
`scripts/make_submission.py` into `dist/submission/`. Upload the CONTENTS of
`dist/submission/` (or `dist/submission.zip`) as the Kaggle agent.
```

- [ ] **Step 2: Write `scripts/make_submission.py`**

```python
"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python). Usage: python scripts/make_submission.py"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
CG = os.path.join(REPO, "pokemon-tcg-ai-battle", "sample_submission",
                  "sample_submission", "cg")
OUT = os.path.join(REPO, "dist", "submission")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py"]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in ("main.py", "deck.csv", "policy.pt", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
    shutil.copytree(CG, os.path.join(OUT, "cg"))
    os.makedirs(os.path.join(OUT, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(OUT, "ptcg", m))
    # self-containment check: import + deck-selection call from a CLEAN cwd/sys.path,
    # with REPO removed from the path so it can only resolve bundled ptcg/cg.
    check = ("import sys; sys.path=[p for p in sys.path if 'Pokemon-Kaggle' not in p "
             "or p.endswith('submission')]; sys.path.insert(0, '.'); "
             "import importlib.util as u; s=u.spec_from_file_location('m','main.py'); "
             "m=u.module_from_spec(s); s.loader.exec_module(m); "
             "d=m.agent({'select':None,'current':None,'logs':[]}); "
             "assert len(d)==60; print('self-contained OK: deck', len(d))")
    r = subprocess.run([sys.executable, "-c", check], cwd=OUT,
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip())
    if r.returncode != 0:
        sys.exit("self-containment check FAILED")
    zip_base = os.path.join(REPO, "dist", "submission")
    shutil.make_archive(zip_base, "zip", OUT)
    print(f"wrote {OUT}/ and {zip_base}.zip")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the assembly**

Run: `python scripts/make_submission.py`
Expected: prints `self-contained OK: deck 60` then `wrote .../dist/submission/ and
.../dist/submission.zip`. The self-containment subprocess proves the bundle resolves
`ptcg`/`cg` from inside itself (repo path stripped). If it fails, a needed module is
missing from `PTCG_MODULES` — add it and re-run.

- [ ] **Step 4: Final smoke against the assembled bundle** (optional but recommended)

Run: `python scripts/test_submission.py 4` after temporarily pointing it at the bundle,
OR trust Task-2's smoke (same `main.py`). Record that the bundle plays legal games.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_submission.py submission_src/README.md
git commit -m "submission bundle assembly and packaging"
```

---

## Self-review notes

- Spec coverage: bundle layout + trimmed `ptcg` set → Task 4; agent flow (deck-return,
  tracker reset, seat, featurize→sample_select) + robustness fallback → Task 1; obs-schema
  compatibility + legality + latency → Task 2 smoke; deck selection round-robin → Task 3;
  self-containment (`PTCG_ENGINE_DIR`, isolated import) → Task 4. Runtime assumption
  (Kaggle torch) and out-of-scope items (greedy A/B, distillation, clock) carried from the
  spec.
- The agent never raises (Task-1 fallback test forces the `except` path); `sample_select`
  already masks to legal options, and the explicit legality guard is belt-and-suspenders.
- `student_config` matching checkpoint-0120's size is verified structurally by the Task-2
  smoke (weights load + real games run); if `load_state_dict` shape-mismatches, Task 2
  fails loudly.
- Big artifacts (`submission_src/policy.pt`, `dist/`) are git-ignored; committed source is
  small (`main.py`, `deck.csv`, `README.md`, scripts, test).
```
