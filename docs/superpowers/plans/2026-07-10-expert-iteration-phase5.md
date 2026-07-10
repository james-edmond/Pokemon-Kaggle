# Expert Iteration (phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the expert-iteration pipeline — search-vs-search self-play recorder, visit-distribution trainer, CI-gated promotion, resumable cycle driver, cloud playbook — per the approved spec `docs/superpowers/specs/2026-07-10-expert-iteration-phase5-design.md`, and validate one miniature cycle end-to-end locally (stage 0).

**Architecture:** Both seats play with the unchanged phase-4 `search_move` under a sims budget; each move records the root's candidate pick-tuples and raw cross-tree visit counts plus featurized/privileged states (phase-2 tensor conventions). A supervised trainer replaces PPO for this loop: policy cross-entropy to the visit distribution, value MSE to outcomes, aux Poisson losses unchanged. A gate script promotes a candidate only when its search-wrapped head-to-head Wilson CI clears 0.50, with anchor non-regression checks. An idempotent driver chains generate→train→gate→promote per cycle.

**Tech Stack:** Python 3.11, torch (CPU locally; CUDA on rented boxes), numpy, the competition's native `cg` engine, pytest, multiprocessing (spawn).

## Global Constraints

- NEVER modify anything under `pokemon-tcg-ai-battle/`.
- The phase-4 shipped behavior must not change: the only production-module edit is Task 1's **additive** `MoveStats` telemetry (new fields default `None`); every existing test in `tests/test_mcts.py` must pass **unmodified**.
- Tests: base python, `python -m pytest tests/<file> -v`, ONE pytest invocation at a time; one battle per process (sequential battles fine, always `close()` in `finally`). Engine RNG is UNSEEDABLE — any new battle-driving test setup must use a fresh-battle retry helper (close-unless-returned pattern; see `tests/test_simsearch.py::_mid_game_session`).
- GPU work uses `venv-train\Scripts\python` locally; plain `python` (base) for tests.
- Commits on `main`, short lowercase messages, **no co-author lines**.
- `runs/`, `champ/`, `dist/`, `submission_src/policy.pt` are gitignored — never `git add` them. EI data/checkpoints live under `runs/ei/` (already covered by `runs/`).
- Long-running local processes (stage-0 loop) are launched by the CONTROLLER or the user, never as children of subagent sessions (phase-2 lesson).
- Windows console: `PYTHONIOENCODING=utf-8` for engine scripts.
- Deployment target is unchanged (Kaggle CPU, phase-4 agent); nothing in this plan touches `submission_src/` or packaging.
- Interfaces you may rely on (verified at HEAD): `Step(player, state, esel, picks, logprob, priv_state)` / `Episode(steps, result, rewards, featurizer_version, decks, collected_seats)` in `ptcg/rollout.py`; `play_league_game(learner, opponent, decks, tables, *, learner_seat, mirror, generator=None, step_cap=5000)` (opponent may be `"random"`); `batched_replay(model, trunk, sb, selb, picks_list) -> (logp, ent)` in `ptcg/replay.py`; `aux_targets(steps, tables, opp_decks) -> (pd, dl, hd)` in `ptcg/ppo.py` (requires `.state`/`.priv_state`/`.player` on steps; `opp_decks` = per-step list of the opponent's 60 ids); `PORTFOLIO` dict + `deck(name)` in `ptcg/decks.py`; `search_move(obs, me, my_deck, tracker, model, tables, session, cfg, rng, gen, tslice) -> (picks|None, MoveStats)` and `SearchConfig(k_trees, sims_per_tree, c_puct, m_multipick)` in `ptcg/mcts.py`; `forced_picks(select)` in `ptcg/clock.py`; `SearchSession` in `ptcg/simsearch.py`; `wilson`-style gates use n≥300 for CI-clears-0.50 promotion.

---

### Task 1: expose root visit counts on `MoveStats` (additive, behavior-identical)

**Files:**
- Modify: `ptcg/mcts.py` (MoveStats, `_vote`, `search_move` vote block)
- Test: `tests/test_mcts.py` (append)

**Interfaces:**
- Consumes: existing `_vote(roots)` and `search_move`.
- Produces: `MoveStats.root_actions: list[tuple] | None`, `MoveStats.root_visits: list[int] | None` (populated iff `searched`); module function `_vote_counts(roots) -> (Counter, Counter)` (votes, wsum). Task 2 relies on `stats.root_actions`/`stats.root_visits`; invariant: `sum(root_visits) == stats.sims`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_mcts.py`:

```python
def test_vote_counts_exposes_summed_visits():
    sess = _FakeSession()
    a = _mk_root(sess)
    b = _mk_root(sess)
    a.N, a.W = [3, 1], [1.0, 0.5]
    b.N, b.W = [1, 3], [0.2, 0.4]
    votes, wsum = M._vote_counts([(a, None), (b, None)])
    assert votes[(0,)] == 4 and votes[(1,)] == 4
    assert abs(wsum[(0,)] - 1.2) < 1e-9 and abs(wsum[(1,)] - 0.9) < 1e-9
    # _vote must agree with the counts it is built on
    assert M._vote([(a, None), (b, None)]) in ((0,), (1,))


def test_movestats_has_root_fields_defaulting_none():
    st = M.MoveStats()
    assert st.root_actions is None and st.root_visits is None
```

And inside the existing `test_search_move_live_engine_legal_and_budgeted`, extend the `if stats.searched:` block with:

```python
            assert stats.root_actions is not None
            assert stats.root_visits is not None
            assert len(stats.root_actions) == len(stats.root_visits)
            assert sum(stats.root_visits) == stats.sims
            assert tuple(picks) in stats.root_actions
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcts.py -v -k "vote_counts or movestats"`
Expected: FAIL with `AttributeError: ... has no attribute '_vote_counts'` / MoveStats TypeError.

- [ ] **Step 3: Implement**

In `ptcg/mcts.py`:

Add the two fields to `MoveStats`:

```python
@dataclass
class MoveStats:
    searched: bool = False
    trees: int = 0
    sims: int = 0
    elapsed: float = 0.0
    reason: str = ""
    root_actions: list = None   # candidate pick-tuples (populated iff searched)
    root_visits: list = None    # summed cross-tree visits, aligned with root_actions
```

Refactor `_vote` through a counts helper (behavior identical):

```python
def _vote_counts(roots):
    """Summed visit and value counters per root pick-tuple across trees."""
    votes, wsum = Counter(), Counter()
    for root, _ in roots:
        if not root.actions:
            continue
        for j, a in enumerate(root.actions):
            votes[a] += root.N[j]
            wsum[a] += root.W[j]
    return votes, wsum


def _vote(roots):
    """Root pick across trees: max summed visits, ties by mean value."""
    votes, wsum = _vote_counts(roots)
    if not votes:
        return None
    return max(votes, key=lambda a: (votes[a],
                                     wsum[a] / votes[a] if votes[a] else float("-inf")))
```

In `search_move`, replace the vote block (`best = _vote(roots)` and its `None` guard) with:

```python
        votes, wsum = _vote_counts(roots)
        if not votes:
            stats.reason = "no-vote"
            return None, stats
        stats.root_actions = list(votes.keys())
        stats.root_visits = [int(votes[a]) for a in stats.root_actions]
        best = max(votes, key=lambda a: (votes[a],
                                         wsum[a] / votes[a] if votes[a] else float("-inf")))
```

(Every `_simulate` from an expanded root appends exactly one `(root, action)` path entry — leaf, terminal, dead-edge, and depth-bound sims alike — so `sum(root_visits) == stats.sims` holds.)

- [ ] **Step 4: Run the whole file**

Run: `python -m pytest tests/test_mcts.py -v`
Expected: all pass (7 existing + 2 new; the live test's new asserts hold).

- [ ] **Step 5: Commit**

```bash
git add ptcg/mcts.py tests/test_mcts.py
git commit -m "expose root visit counts on movestats for expert iteration"
```

---

### Task 2: `ptcg/selfplay_search.py` — search-vs-search recorder

**Files:**
- Create: `ptcg/selfplay_search.py`
- Test: `tests/test_selfplay_search.py`

**Interfaces:**
- Consumes: Task 1's `stats.root_actions/root_visits`; `search_move`, `SearchConfig`, `SearchSession`, `forced_picks`, `sample_select`, `featurize_state/encode_select/featurize_privileged/FEATURIZER_VERSION`, `BeliefTracker`, `BattleSession`, `ptcg.decks.PORTFOLIO/deck`.
- Produces: `@dataclass EIStep(player, state, esel, priv_state, actions=None, visits=None)`; `@dataclass EIGame(steps, result, rewards, decks, featurizer_version=FEATURIZER_VERSION)`; `sample_deck_pair(rng, mirror_frac=0.3) -> (name, name)`; `play_search_game(net0, net1, deck_names, tables, *, cfg, rng, gen, session=None, record=True, step_cap=5000) -> EIGame`. `EIStep` keeps field names `state`/`priv_state`/`player` so `ppo.aux_targets` consumes it unchanged. `record=False` skips privileged featurization and step recording (gates use it; `steps == []`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_selfplay_search.py`:

```python
import random

import torch

from ptcg.cards import build_tables
from ptcg.decks import PORTFOLIO
from ptcg.mcts import SearchConfig
from ptcg.selfplay_search import (EIGame, EIStep, play_search_game,
                                  sample_deck_pair)


def _tiny_net(tables):
    import os

    from ptcg.model import PolicyModel, student_config
    m = PolicyModel(student_config(tables))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m.load_state_dict(torch.load(
        os.path.join(repo, "submission_src", "policy.pt"),
        map_location="cpu", weights_only=True))
    m.eval()
    return m


def test_sample_deck_pair_mirror_and_distinct():
    rng = random.Random(0)
    pairs = [sample_deck_pair(rng) for _ in range(200)]
    names = set(PORTFOLIO)
    assert all(a in names and b in names for a, b in pairs)
    mirrors = sum(1 for a, b in pairs if a == b)
    assert 20 <= mirrors <= 120          # ~30% of 200, loose bounds
    assert any(a != b for a, b in pairs)


def test_play_search_game_records_targets_and_outcome():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=2, sims_per_tree=8)
    rng = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=rng, gen=gen)
    assert isinstance(g, EIGame)
    assert g.result in (0, 1, 2)
    assert g.rewards in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
    assert len(g.decks) == 2 and len(g.decks[0]) == 60
    assert len(g.steps) > 0
    searched = [s for s in g.steps if s.actions is not None]
    valueonly = [s for s in g.steps if s.actions is None]
    assert searched, "no move recorded a visit distribution"
    for s in g.steps:
        assert isinstance(s, EIStep)
        assert s.player in (0, 1)
        assert s.priv_state is not None
        if s.actions is not None:
            assert len(s.actions) == len(s.visits) >= 2
            assert all(isinstance(a, tuple) for a in s.actions)
            assert sum(s.visits) > 0
    # aux_targets consumes EIStep unchanged (field-name contract)
    from ptcg.ppo import aux_targets
    opp = [g.decks[1 - s.player] for s in g.steps]
    pd, dl, hd = aux_targets(g.steps, tables, opp)
    assert pd.shape[0] == len(g.steps) and dl.shape[0] == len(g.steps)
    assert valueonly is not None   # trivial/forced moves may or may not occur


def test_play_search_game_record_false_skips_steps():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=1, sims_per_tree=4)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=random.Random(5),
                         gen=torch.Generator().manual_seed(5), record=False)
    assert g.steps == [] and g.result in (0, 1, 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_selfplay_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.selfplay_search'`

- [ ] **Step 3: Implement**

Create `ptcg/selfplay_search.py`:

```python
"""Search-vs-search self-play with expert-iteration recording.

Both seats pick moves with the phase-4 search under a SIMS budget
(tslice=inf so k_trees*sims_per_tree binds): data strength is
machine-load-independent. Each recorded move stores the root's candidate
pick-tuples and RAW cross-tree visit counts (the trainer normalizes, so
temperature stays a train-time knob), plus the public/privileged
featurizations in phase-2 tensor conventions. Moves without a usable
distribution (forced picks, single-action shortcuts, search fallbacks)
are recorded value-only (actions=None): every state still trains the
value and aux heads. record=False skips recording entirely (gate games).
"""
import random
from dataclasses import dataclass

import torch

from .action import sample_select
from .clock import forced_picks
from .engine import BattleSession
from .featurize import (FEATURIZER_VERSION, encode_select,
                        featurize_privileged, featurize_state)
from .mcts import search_move
from .simsearch import SearchSession
from .tracker import BeliefTracker


@dataclass
class EIStep:
    player: int
    state: object            # TokenizedState (public, acting seat)
    esel: object             # EncodedSelect
    priv_state: object       # TokenizedState (privileged)
    actions: list = None     # list[tuple[int,...]] root candidates, or None
    visits: list = None      # raw visit counts aligned with actions, or None


@dataclass
class EIGame:
    steps: list
    result: int
    rewards: tuple
    decks: tuple
    featurizer_version: int = FEATURIZER_VERSION


def sample_deck_pair(rng, mirror_frac=0.3):
    """Two portfolio deck names: mirror with prob mirror_frac, else distinct."""
    from .decks import PORTFOLIO
    names = sorted(PORTFOLIO)
    a = rng.choice(names)
    if rng.random() < mirror_frac:
        return (a, a)
    return (a, rng.choice([n for n in names if n != a]))


def play_search_game(net0, net1, deck_names, tables, *, cfg, rng, gen,
                     session=None, record=True, step_cap=5000):
    """One battle where seat 0 plays net0 and seat 1 plays net1, both with
    search. Returns an EIGame (steps empty when record=False)."""
    from .decks import deck as deck_by_name
    decks = (deck_by_name(deck_names[0]), deck_by_name(deck_names[1]))
    nets = (net0, net1)
    session = session or SearchSession()
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
            ts = es = pv = None
            if record:
                ts = featurize_state(s.obs, me, decks[me],
                                     trackers[me].snapshot(), tables)
                es = encode_select(s.obs, ts, tables)
                # a seat that has not yet acted has no obs of its own: its
                # slot in last_obs holds the other seat's obs, where its hand
                # is None. Source that hand from VisualizeData.
                vcur = s.viz_current()
                viz_hands = None
                if not (seen[0] and seen[1]):
                    vp = vcur.get("players") or []
                    if len(vp) == 2:
                        viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
                pv = featurize_privileged(last_obs[0], last_obs[1], decks,
                                          tables, viz=vcur,
                                          viz_hands=viz_hands)
            actions = visits = None
            fp = forced_picks(s.obs["select"])
            if fp is not None:
                picks = fp
            else:
                picks, st = search_move(
                    s.obs, me, decks[me], trackers[me], nets[me], tables,
                    session, cfg, rng, gen, tslice=float("inf"))
                if picks is None:
                    if ts is None:
                        ts = featurize_state(s.obs, me, decks[me],
                                             trackers[me].snapshot(), tables)
                        es = encode_select(s.obs, ts, tables)
                    d = sample_select(nets[me], ts, es, gen)
                    picks = d.picks
                elif (st.searched and st.root_actions is not None
                        and len(st.root_actions) >= 2
                        and sum(st.root_visits) > 0):
                    actions = list(st.root_actions)
                    visits = [int(v) for v in st.root_visits]
            if record:
                steps.append(EIStep(me, ts, es, pv, actions, visits))
            s.select(list(picks))
        r = s.result
    finally:
        s.close()
    rewards = (0.0, 0.0) if r == 2 else ((1.0, -1.0) if r == 0 else (-1.0, 1.0))
    return EIGame(steps, r, rewards, decks=(list(decks[0]), list(decks[1])))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_selfplay_search.py -v`
Expected: 3 passed (the game test runs a real battle with tiny sims — a couple of minutes is normal). If the game test flakes on a degenerate battle, harden the SETUP only (retry loop constructing a fresh game via a different rng seed, close-unless-returned), never the assertions.

- [ ] **Step 5: Commit**

```bash
git add ptcg/selfplay_search.py tests/test_selfplay_search.py
git commit -m "add search self-play recorder with visit-distribution targets"
```

---

### Task 3: `scripts/gen_ei.py` — generation workers + manifest

**Files:**
- Create: `scripts/gen_ei.py`

**Interfaces:**
- Consumes: Task 2's `play_search_game/sample_deck_pair/EIGame`; `SearchConfig`.
- Produces: CLI `python scripts/gen_ei.py --ckpt <path> --out <dir> --games N --workers W --k K --sims S --seed X [--batch 25] [--mirror-frac 0.3]`. Writes `worker-<i>-batch-<j>.pt` files (each a `list[EIGame]` via `torch.save`) and, on completion, `manifest.json` `{"games": N, "moves": M, "args": {...}}`. The manifest's existence is the completeness marker Tasks 6/8 rely on. Also produces `load_policy(path, tables)` convention: accepts a bare policy `state_dict` OR a training checkpoint dict with a `"policy"` key.

- [ ] **Step 1: Write the script**

Create `scripts/gen_ei.py`:

```python
"""Generate expert-iteration self-play data (search vs search, one net).
Usage:
  python scripts/gen_ei.py --ckpt submission_src/policy.pt \
      --out runs/ei/dev/cycle-0/data --games 80 --workers 3 --k 2 --sims 16 --seed 1
Writes worker-<i>-batch-<j>.pt (list[EIGame]) and manifest.json when complete.
Idempotent-ish: refuses to run if manifest.json already exists."""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_policy(path, tables):
    """Bare policy state_dict OR training checkpoint with a 'policy' key."""
    import torch

    from ptcg.model import PolicyModel, student_config
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob
    m = PolicyModel(student_config(tables))
    m.load_state_dict(sd)
    m.eval()
    return m


def _worker(args):
    (wid, games, ckpt, out, k, sims, seed, batch, mirror_frac) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.mcts import SearchConfig
    from ptcg.selfplay_search import play_search_game, sample_deck_pair
    from ptcg.simsearch import SearchSession
    tables = build_tables()
    net = load_policy(ckpt, tables)
    cfg = SearchConfig(k_trees=k, sims_per_tree=sims)
    rng = random.Random(seed * 1000 + wid)
    gen = torch.Generator().manual_seed(seed * 1000 + wid)
    session = SearchSession()   # ONE arena per worker (no native free exists)
    buf, bi, moves, done = [], 0, 0, 0
    for g in range(games):
        names = sample_deck_pair(rng, mirror_frac)
        game = play_search_game(net, net, names, tables, cfg=cfg, rng=rng,
                                gen=gen, session=session)
        buf.append(game)
        moves += len(game.steps)
        done += 1
        if len(buf) >= batch:
            torch.save(buf, os.path.join(out, f"worker-{wid}-batch-{bi}.pt"))
            buf, bi = [], bi + 1
    if buf:
        torch.save(buf, os.path.join(out, f"worker-{wid}-batch-{bi}.pt"))
    return {"games": done, "moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--mirror-frac", type=float, default=0.3)
    a = ap.parse_args()
    mani = os.path.join(a.out, "manifest.json")
    if os.path.exists(mani):
        sys.exit(f"refusing: {mani} already exists (complete run)")
    os.makedirs(a.out, exist_ok=True)
    per = a.games // a.workers
    rem = a.games - per * a.workers
    jobs = [(w, per + (1 if w < rem else 0), a.ckpt, a.out, a.k, a.sims,
             a.seed, a.batch, a.mirror_frac)
            for w in range(a.workers) if per + (1 if w < rem else 0) > 0]
    if len(jobs) == 1:
        stats = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(_worker, jobs)
    total = {"games": sum(s["games"] for s in stats),
             "moves": sum(s["moves"] for s in stats),
             "args": vars(a)}
    with open(mani, "w") as f:
        json.dump(total, f, indent=2)
    print(f"gen_ei: {total['games']} games, {total['moves']} moves -> {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke it (tiny)**

Run: `PYTHONIOENCODING=utf-8 python scripts/gen_ei.py --ckpt submission_src/policy.pt --out runs/ei/dev-smoke/data --games 4 --workers 2 --k 1 --sims 6 --seed 1 --batch 2`
Expected: completes in a few minutes; prints `gen_ei: 4 games, <M> moves -> runs/ei/dev-smoke/data`; the dir contains `worker-*.pt` files + `manifest.json`. Re-running the same command exits with `refusing: ... already exists`.

- [ ] **Step 3: Verify the data loads**

Run: `python -c "import torch,glob; gs=[g for f in glob.glob('runs/ei/dev-smoke/data/worker-*.pt') for g in torch.load(f,weights_only=False)]; print(len(gs),'games', sum(len(g.steps) for g in gs),'moves', sum(1 for g in gs for s in g.steps if s.actions))"`
Expected: `4 games <M> moves <K>` with K ≥ 1 (some searched moves recorded).

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_ei.py
git commit -m "add ei data generation workers with completion manifest"
```

---

### Task 4: `ptcg/ei.py` part 1 — targets + single-pick policy loss + wilson

**Files:**
- Create: `ptcg/ei.py`
- Test: `tests/test_ei.py`

**Interfaces:**
- Consumes: `ppo.aux_targets`, `model.collate_states/collate_selects`, `model.option_logits/public_value/prize_diff/aux_decklist/aux_hand`, Task 2's `EIStep/EIGame`.
- Produces (Tasks 5-7 rely on these exact names):
  - `wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `@dataclass EIConfig(lr=2e-4, epochs=2, minibatch=128, pi_temp=1.0, kl_coef=0.0, vf_coef=1.0, aux_coef=0.1, grad_clip=1.0, device="cpu", seed=0)`
  - `flatten_games(games) -> list[tuple[EIStep, float, list]]` — (step, z from acting seat, opponent 60-card deck)
  - `is_single_pick(step) -> bool` — every candidate tuple has len ≤ 1
  - `pi_targets_single(steps, O, temp) -> torch.Tensor[B, O+1]` — normalized visit targets; a `()` candidate maps to the done column O
  - `single_pick_loss(policy, steps, zs, opp_decks, tables, cfg, incumbent=None) -> (loss, parts_dict)` — fused policy CE + value MSE + aux Poisson (+ optional KL) on one collated batch

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ei.py`:

```python
import math
import random

import torch

from ptcg.cards import build_tables
from ptcg.ei import (EIConfig, flatten_games, is_single_pick,
                     pi_targets_single, single_pick_loss, wilson)
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import encode_select, featurize_state, featurize_privileged
from ptcg.selfplay_search import EIGame, EIStep
from ptcg.tracker import BeliefTracker


def test_wilson_matches_known_value():
    lo, hi = wilson(4, 12)
    assert abs(lo - 0.138) < 2e-3 and abs(hi - 0.609) < 2e-3


def _fabricated_game(tables, n_states=6):
    """Real featurized states from a live battle; fabricated candidates."""
    deck = load_sample_deck()
    for t in range(6):
        s = BattleSession(deck, deck)
        keep = False
        try:
            rng = random.Random(11 + t)
            trk = {0: BeliefTracker(0), 1: BeliefTracker(1)}
            steps = []
            for _ in range(60):
                if s.done or len(steps) >= n_states:
                    break
                me = s.obs["current"]["yourIndex"]
                trk[me].update(s.obs.get("logs") or [])
                sel = s.obs["select"]
                ts = featurize_state(s.obs, me, deck, trk[me].snapshot(),
                                     tables)
                es = encode_select(s.obs, ts, tables)
                pv = featurize_privileged(s.obs, s.obs, (deck, deck), tables)
                n = len(sel["option"])
                if sel["maxCount"] == 1 and n >= 2:
                    acts = [(j,) for j in range(min(n, 3))]
                    vis = [3, 1] + ([1] if len(acts) == 3 else [])
                    steps.append(EIStep(me, ts, es, pv, acts, vis))
                else:
                    steps.append(EIStep(me, ts, es, pv, None, None))
                s.select(random_picks(s.obs, rng))
            if len(steps) >= 3 and any(x.actions for x in steps):
                keep = True
                return EIGame(steps, 0, (1.0, -1.0), (list(deck), list(deck)))
        finally:
            s.close()
    raise AssertionError("no usable fabricated game")


def test_flatten_and_single_pick_partition():
    tables = build_tables()
    g = _fabricated_game(tables)
    flat = flatten_games([g])
    assert len(flat) == len(g.steps)
    st, z, od = flat[0]
    assert z == g.rewards[st.player]
    assert od == g.decks[1 - st.player]
    singles = [s for s, _, _ in flat if s.actions and is_single_pick(s)]
    assert singles


def test_pi_targets_temperature_math():
    tables = build_tables()
    g = _fabricated_game(tables)
    s = next(x for x in g.steps if x.actions and len(x.actions) == 2)
    t1 = pi_targets_single([s], O=8, temp=1.0)
    assert abs(float(t1[0, s.actions[0][0]]) - 0.75) < 1e-6   # visits [3,1]
    assert abs(float(t1[0, s.actions[1][0]]) - 0.25) < 1e-6
    t2 = pi_targets_single([s], O=8, temp=0.5)                # ^(1/T)=^2
    assert abs(float(t2[0, s.actions[0][0]]) - 0.9) < 1e-6    # 9:1
    assert abs(float(t2.sum()) - 1.0) < 1e-6


def test_single_pick_loss_decreases_with_training():
    import os

    from ptcg.model import PolicyModel, tiny_config
    tables = build_tables()
    torch.manual_seed(0)
    net = PolicyModel(tiny_config(tables))
    g = _fabricated_game(tables)
    flat = [x for x in flatten_games([g]) if x[0].actions
            and is_single_pick(x[0])]
    steps = [x[0] for x in flat]
    zs = [x[1] for x in flat]
    ods = [x[2] for x in flat]
    cfg = EIConfig(lr=5e-3, device="cpu")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    first = last = None
    for i in range(30):
        loss, parts = single_pick_loss(net, steps, zs, ods, tables, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, (first, last)
    assert set(parts) >= {"loss_pi", "loss_v", "loss_aux"}
    assert all(math.isfinite(v) for v in parts.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_ei.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.ei'`

- [ ] **Step 3: Implement**

Create `ptcg/ei.py`:

```python
"""Expert-iteration training: supervised losses from search self-play.

Policy: cross-entropy to the normalized root visit distribution (single-
pick selects fuse policy+value+aux on one collated trunk; multi-pick
candidates replay through batched_replay). Value: MSE to the game outcome
from the acting seat, on EVERY state including turn-starts (recalibrates
the phase-4-diagnosed turn-phase artifact). Aux heads keep their phase-2
Poisson/MSE targets — the determinizer's accuracy is a search input.
"""
import math
import random
from dataclasses import dataclass

import torch

from .ppo import aux_targets


def wilson(w, n, z=1.96):
    """Wilson 95% score interval (lo, hi) for w wins of n games."""
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


@dataclass
class EIConfig:
    lr: float = 2e-4
    epochs: int = 2
    minibatch: int = 128
    pi_temp: float = 1.0
    kl_coef: float = 0.0
    vf_coef: float = 1.0
    aux_coef: float = 0.1
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 0


def flatten_games(games):
    """[(step, z_for_acting_seat, opponent_60_card_deck), ...]"""
    out = []
    for g in games:
        for s in g.steps:
            out.append((s, float(g.rewards[s.player]),
                        list(g.decks[1 - s.player])))
    return out


def is_single_pick(step):
    return all(len(a) <= 1 for a in step.actions)


def pi_targets_single(steps, O, temp):
    """[B, O+1] normalized visit targets; () maps to the done column O."""
    t = torch.zeros(len(steps), O + 1)
    for i, s in enumerate(steps):
        w = torch.tensor([float(v) for v in s.visits])
        if temp != 1.0:
            w = w.clamp(min=1e-9) ** (1.0 / temp)
        w = w / w.sum()
        for a, p in zip(s.actions, w.tolist()):
            col = a[0] if len(a) == 1 else O
            t[i, col] += p
    return t


def _aux_loss(policy, trunk, steps, opp_decks, tables, device):
    poiss = torch.nn.PoissonNLLLoss(log_input=False, full=False)
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_decks)
    pd_t, dl_t, hd_t = pd_t.to(device), dl_t.to(device), hd_t.to(device)
    return (((policy.prize_diff(trunk) - pd_t) ** 2).mean()
            + poiss(policy.aux_decklist(trunk), dl_t)
            + poiss(policy.aux_hand(trunk), hd_t))


def single_pick_loss(policy, steps, zs, opp_decks, tables, cfg,
                     incumbent=None):
    """Fused policy CE + value MSE + aux losses on one collated batch."""
    from .model import collate_selects, collate_states
    dev = torch.device(cfg.device)
    sb = {k: v.to(dev) for k, v in collate_states(
        [s.state for s in steps]).items()}
    selb = {k: v.to(dev) for k, v in collate_selects(
        [s.esel for s in steps]).items()}
    B, O = selb["opt_type"].shape
    picked = torch.zeros((B, O + 1), dtype=torch.bool, device=dev)
    trunk = policy.encode(sb)
    logits = policy.option_logits(trunk, sb, selb, picked)
    logp = torch.log_softmax(logits, dim=-1)
    targets = pi_targets_single(steps, O, cfg.pi_temp).to(dev)
    # candidates are legal by construction, so target mass never sits on a
    # -inf column; nan_to_num guards the 0 * -inf corner on masked columns
    loss_pi = -(targets * torch.nan_to_num(logp, neginf=0.0)).sum(-1).mean()
    v = policy.public_value(trunk)
    z_t = torch.tensor(zs, dtype=torch.float32, device=dev)
    loss_v = ((v - z_t) ** 2).mean()
    loss_aux = _aux_loss(policy, trunk, steps, opp_decks, tables, dev)
    parts = {"loss_pi": float(loss_pi), "loss_v": float(loss_v),
             "loss_aux": float(loss_aux)}
    loss = loss_pi + cfg.vf_coef * loss_v + cfg.aux_coef * loss_aux
    if incumbent is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            it = incumbent.encode(sb)
            il = incumbent.option_logits(it, sb, selb, picked)
            q = torch.softmax(il, dim=-1)
        p = torch.softmax(logits, dim=-1)
        kl = (p * (torch.nan_to_num(torch.log(p.clamp(min=1e-9)), neginf=0.0)
                   - torch.nan_to_num(torch.log(q.clamp(min=1e-9)),
                                      neginf=0.0))).sum(-1).mean()
        loss = loss + cfg.kl_coef * kl
        parts["kl"] = float(kl)
    return loss, parts
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ei.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ptcg/ei.py tests/test_ei.py
git commit -m "add ei targets, wilson ci, and fused single-pick loss"
```

---

### Task 5: `ptcg/ei.py` part 2 — multi-pick loss, value-only batches, `train_ei`

**Files:**
- Modify: `ptcg/ei.py` (append)
- Test: `tests/test_ei.py` (append)

**Interfaces:**
- Consumes: Task 4's pieces; `replay.batched_replay`.
- Produces: `multi_pick_loss(policy, step, z, opp_deck, tables, cfg) -> (loss, parts)`; `value_only_loss(policy, steps, zs, opp_decks, tables, cfg) -> (loss, parts)`; `train_ei(policy, games, tables, cfg, incumbent=None) -> dict` (metrics: `loss_pi, loss_v, loss_aux, n_single, n_multi, n_valueonly, epochs_ran`; shuffles with `cfg.seed`; minibatches single-pick states; per-state multi-pick; value-only minibatched; AdamW with `cfg.grad_clip`).

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_ei.py`:

```python
def test_multi_pick_loss_matches_manual_logprob_weighting():
    import os

    from ptcg.model import PolicyModel, tiny_config
    from ptcg.action import replay_logprob
    from ptcg.ei import multi_pick_loss
    tables = build_tables()
    torch.manual_seed(1)
    net = PolicyModel(tiny_config(tables))
    g = _fabricated_game(tables)
    s = next(x for x in g.steps if x.actions and len(x.actions) >= 2)
    cfg = EIConfig()
    loss, parts = multi_pick_loss(net, s, 1.0, g.decks[1 - s.player],
                                  tables, cfg)
    # manual: -sum(pi_a * logp_a) via the B==1 replay path
    with torch.no_grad():
        lps = replay_logprob(net, [s.state] * len(s.actions),
                             [s.esel] * len(s.actions),
                             [list(a) for a in s.actions])
    w = torch.tensor([float(v) for v in s.visits])
    pi = w / w.sum()
    manual = -(pi * lps).sum()
    assert abs(float(parts["loss_pi"]) - float(manual)) < 1e-4


def test_train_ei_runs_and_improves_on_fabricated_data():
    from ptcg.model import PolicyModel, tiny_config
    from ptcg.ei import train_ei
    tables = build_tables()
    torch.manual_seed(2)
    net = PolicyModel(tiny_config(tables))
    games = [_fabricated_game(tables) for _ in range(2)]
    cfg = EIConfig(lr=3e-3, epochs=4, minibatch=8, device="cpu", seed=0)
    m1 = train_ei(net, games, tables, cfg)
    m2 = train_ei(net, games, tables, cfg)
    assert m2["loss_pi"] < m1["loss_pi"]
    assert m1["n_single"] + m1["n_multi"] + m1["n_valueonly"] == sum(
        len(g.steps) for g in games)
    for k in ("loss_pi", "loss_v", "loss_aux"):
        assert k in m1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_ei.py -v -k "multi_pick or train_ei"`
Expected: FAIL with ImportError (`multi_pick_loss`/`train_ei` not defined).

- [ ] **Step 3: Implement (append to `ptcg/ei.py`)**

```python
def multi_pick_loss(policy, step, z, opp_deck, tables, cfg):
    """-sum(pi_a * logprob(sequence a)) for one multi-pick state, plus
    value/aux on the same trunk (row 0)."""
    from .model import collate_selects, collate_states
    from .replay import batched_replay
    dev = torch.device(cfg.device)
    n = len(step.actions)
    sb = {k: v.to(dev) for k, v in collate_states(
        [step.state] * n).items()}
    selb = {k: v.to(dev) for k, v in collate_selects(
        [step.esel] * n).items()}
    trunk = policy.encode(sb)
    logp, _ = batched_replay(policy, trunk, sb, selb,
                             [list(a) for a in step.actions])
    w = torch.tensor([float(v) for v in step.visits], device=dev)
    if cfg.pi_temp != 1.0:
        w = w.clamp(min=1e-9) ** (1.0 / cfg.pi_temp)
    pi = w / w.sum()
    loss_pi = -(pi * logp).sum()
    v = policy.public_value(trunk[0:1])
    loss_v = (v - torch.tensor([z], device=dev)) ** 2
    loss_aux = _aux_loss(policy, trunk[0:1], [step], [opp_deck], tables, dev)
    loss = loss_pi + cfg.vf_coef * loss_v.mean() + cfg.aux_coef * loss_aux
    return loss, {"loss_pi": float(loss_pi), "loss_v": float(loss_v.mean()),
                  "loss_aux": float(loss_aux)}


def value_only_loss(policy, steps, zs, opp_decks, tables, cfg):
    """Value + aux losses for states without a policy target."""
    from .model import collate_states
    dev = torch.device(cfg.device)
    sb = {k: v.to(dev) for k, v in collate_states(
        [s.state for s in steps]).items()}
    trunk = policy.encode(sb)
    v = policy.public_value(trunk)
    z_t = torch.tensor(zs, dtype=torch.float32, device=dev)
    loss_v = ((v - z_t) ** 2).mean()
    loss_aux = _aux_loss(policy, trunk, steps, opp_decks, tables, dev)
    loss = cfg.vf_coef * loss_v + cfg.aux_coef * loss_aux
    return loss, {"loss_v": float(loss_v), "loss_aux": float(loss_aux)}


def train_ei(policy, games, tables, cfg, incumbent=None):
    """One training pass (cfg.epochs) over the games. Returns metrics."""
    dev = torch.device(cfg.device)
    policy.to(dev)
    policy.train()
    if incumbent is not None:
        incumbent.to(dev)
        incumbent.eval()
    flat = flatten_games(games)
    singles = [(s, z, od) for s, z, od in flat
               if s.actions is not None and is_single_pick(s)]
    multis = [(s, z, od) for s, z, od in flat
              if s.actions is not None and not is_single_pick(s)]
    vonly = [(s, z, od) for s, z, od in flat if s.actions is None]
    rng = random.Random(cfg.seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    agg = {"loss_pi": 0.0, "loss_v": 0.0, "loss_aux": 0.0}
    n_pi = n_va = 0
    for _ in range(cfg.epochs):
        rng.shuffle(singles)
        rng.shuffle(vonly)
        for lo in range(0, len(singles), cfg.minibatch):
            batch = singles[lo:lo + cfg.minibatch]
            loss, parts = single_pick_loss(
                policy, [b[0] for b in batch], [b[1] for b in batch],
                [b[2] for b in batch], tables, cfg, incumbent=incumbent)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_pi"] += parts["loss_pi"]
            agg["loss_v"] += parts["loss_v"]
            agg["loss_aux"] += parts["loss_aux"]
            n_pi += 1
            n_va += 1
        for s, z, od in multis:
            loss, parts = multi_pick_loss(policy, s, z, od, tables, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_pi"] += parts["loss_pi"]
            n_pi += 1
        for lo in range(0, len(vonly), cfg.minibatch):
            batch = vonly[lo:lo + cfg.minibatch]
            loss, parts = value_only_loss(
                policy, [b[0] for b in batch], [b[1] for b in batch],
                [b[2] for b in batch], tables, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.grad_clip)
            opt.step()
            agg["loss_v"] += parts["loss_v"]
            agg["loss_aux"] += parts["loss_aux"]
            n_va += 1
    policy.eval()
    return {"loss_pi": agg["loss_pi"] / max(n_pi, 1),
            "loss_v": agg["loss_v"] / max(n_va, 1),
            "loss_aux": agg["loss_aux"] / max(n_va, 1),
            "n_single": len(singles), "n_multi": len(multis),
            "n_valueonly": len(vonly), "epochs_ran": cfg.epochs}
```

- [ ] **Step 4: Run the whole test file**

Run: `python -m pytest tests/test_ei.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ptcg/ei.py tests/test_ei.py
git commit -m "add multi-pick and value-only losses and the ei trainer"
```

---

### Task 6: `scripts/train_ei.py` — CLI trainer with replay mixing

**Files:**
- Create: `scripts/train_ei.py`

**Interfaces:**
- Consumes: `ei.EIConfig/train_ei`, `gen_ei.load_policy` (import from `scripts` via path insert — copy the 8-line loader instead: scripts must stay independently runnable; define `_load_policy` locally, same semantics).
- Produces: CLI `python scripts/train_ei.py --data <dir> [--replay <dir> ...] --ckpt-in <path> --ckpt-out <path> [--replay-ratio 0.25] [--device cpu] [--lr 2e-4] [--epochs 2] [--minibatch 128] [--pi-temp 1.0] [--kl-coef 0.0] [--seed 0]`. Saves `{"policy": state_dict, "ei_config": {...}, "metrics": {...}}` to `--ckpt-out` (a dict WITH a `"policy"` key — loadable by `load_policy` everywhere). Refuses to overwrite an existing `--ckpt-out`.

- [ ] **Step 1: Write the script**

Create `scripts/train_ei.py`:

```python
"""Train one expert-iteration cycle from recorded search self-play data.
Usage:
  python scripts/train_ei.py --data runs/ei/dev/cycle-0/data \
      --ckpt-in submission_src/policy.pt --ckpt-out runs/ei/dev/cycle-0/ckpt.pt \
      [--replay runs/ei/dev/cycle--1/data] [--replay-ratio 0.25] [--kl-coef 0.02]
Replay mixing: appends whole games from --replay dirs (oldest-first order
as globbed) until replay moves ~= replay-ratio * fresh moves."""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_policy(path, tables):
    import torch

    from ptcg.model import PolicyModel, student_config
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob
    m = PolicyModel(student_config(tables))
    m.load_state_dict(sd)
    m.eval()
    return m


def _load_games(d):
    import torch
    out = []
    for f in sorted(glob.glob(os.path.join(d, "worker-*.pt"))):
        out.extend(torch.load(f, weights_only=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay", action="append", default=[])
    ap.add_argument("--replay-ratio", type=float, default=0.25)
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--pi-temp", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.ckpt_out):
        sys.exit(f"refusing: {a.ckpt_out} already exists")
    import torch

    from ptcg.cards import build_tables
    from ptcg.ei import EIConfig, train_ei
    tables = build_tables()
    policy = _load_policy(a.ckpt_in, tables)
    incumbent = _load_policy(a.ckpt_in, tables) if a.kl_coef > 0 else None
    games = _load_games(a.data)
    fresh_moves = sum(len(g.steps) for g in games)
    target = int(fresh_moves * a.replay_ratio)
    got = 0
    for rd in a.replay:
        for g in _load_games(rd):
            if got >= target:
                break
            games.append(g)
            got += len(g.steps)
    cfg = EIConfig(lr=a.lr, epochs=a.epochs, minibatch=a.minibatch,
                   pi_temp=a.pi_temp, kl_coef=a.kl_coef, device=a.device,
                   seed=a.seed)
    metrics = train_ei(policy, games, tables, cfg, incumbent=incumbent)
    os.makedirs(os.path.dirname(a.ckpt_out) or ".", exist_ok=True)
    torch.save({"policy": policy.state_dict(),
                "ei_config": vars(a), "metrics": metrics}, a.ckpt_out)
    print("train_ei:", json.dumps(metrics))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke it on the Task-3 data**

Run: `PYTHONIOENCODING=utf-8 python scripts/train_ei.py --data runs/ei/dev-smoke/data --ckpt-in submission_src/policy.pt --ckpt-out runs/ei/dev-smoke/ckpt.pt --epochs 1 --minibatch 32`
Expected: prints `train_ei: {"loss_pi": ..., "loss_v": ..., ...}` with finite numbers; `runs/ei/dev-smoke/ckpt.pt` exists. Re-run → `refusing: ... already exists`.

- [ ] **Step 3: Verify the checkpoint round-trips**

Run: `python -c "import sys,os; sys.path.insert(0,'.'); from ptcg.cards import build_tables; sys.path.insert(0,'scripts'); from train_ei import _load_policy; m=_load_policy('runs/ei/dev-smoke/ckpt.pt', build_tables()); print('loaded ok', sum(p.numel() for p in m.parameters()))"`
Expected: `loaded ok <param-count>`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ei.py
git commit -m "add ei cycle trainer cli with replay mixing"
```

---

### Task 7: `scripts/gate_ei.py` + `scripts/audit_value_phase.py`

**Files:**
- Create: `scripts/gate_ei.py`
- Create: `scripts/audit_value_phase.py`

**Interfaces:**
- Consumes: `ei.wilson`, `selfplay_search.play_search_game/sample_deck_pair` (with `record=False`), `rollout.play_league_game`, `SearchConfig`, the `_load_policy` convention (local copy).
- Produces: CLI `python scripts/gate_ei.py --candidate <ckpt> --incumbent <ckpt> --out <gate.json> [--games-search 300] [--games-raw 60] [--anchor champ/phase3-generalist-r120.pt] [--anchor-games 60] [--k 3] [--sims 24] [--workers 3] [--seed 7]`. Writes JSON: `{"search_h2h": {"wins", "games", "wr", "lo", "hi"}, "raw_h2h": {...}, "vs_random": {...}, "vs_anchor_search": {...}, "promote": bool}`. **Promotion rule (pre-registered): `search_h2h.lo > 0.50` AND `vs_random.wr >= 0.85` AND `vs_anchor_search.wr >= 0.55`.** `audit_value_phase.py --ckpt <path> [--games 6] --out <json>`: non-gating turn-boundary calibration statistic.

- [ ] **Step 1: Write the gate script**

Create `scripts/gate_ei.py`:

```python
"""Promotion gate: candidate vs incumbent. Pre-registered rule:
PROMOTE iff search-wrapped head-to-head Wilson lo > 0.50 (n>=300 default)
AND raw candidate beats random >= 0.85 AND candidate-with-search beats the
frozen phase-3 anchor-with-search >= 0.55. Workers run whole games
sequentially (one battle per process); spawn pool."""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_policy(path, tables):
    import torch

    from ptcg.model import PolicyModel, student_config
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob
    m = PolicyModel(student_config(tables))
    m.load_state_dict(sd)
    m.eval()
    return m


def _search_chunk(args):
    (games, seed, cand, opp, k, sims) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.mcts import SearchConfig
    from ptcg.selfplay_search import play_search_game, sample_deck_pair
    from ptcg.simsearch import SearchSession
    tables = build_tables()
    a = _load_policy(cand, tables)
    b = _load_policy(opp, tables)
    cfg = SearchConfig(k_trees=k, sims_per_tree=sims)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    session = SearchSession()   # ONE arena per worker (no native free exists)
    wins = 0
    for g in range(games):
        names = sample_deck_pair(rng)
        seat_a = g % 2
        nets = (a, b) if seat_a == 0 else (b, a)
        game = play_search_game(nets[0], nets[1], names, tables, cfg=cfg,
                                rng=rng, gen=gen, record=False,
                                session=session)
        if game.result == seat_a:
            wins += 1
    return {"wins": wins, "games": games}


def _raw_chunk(args):
    (games, seed, cand, opp_spec) = args
    import torch

    from ptcg.cards import build_tables
    from ptcg.rollout import play_league_game
    from ptcg.selfplay_search import sample_deck_pair
    from ptcg.decks import deck as deck_by_name
    tables = build_tables()
    a = _load_policy(cand, tables)
    opp = "random" if opp_spec == "random" else _load_policy(opp_spec, tables)
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    wins = 0
    for g in range(games):
        names = sample_deck_pair(rng)
        decks = (deck_by_name(names[0]), deck_by_name(names[1]))
        seat_a = g % 2
        ep = play_league_game(a, opp, decks, tables, learner_seat=seat_a,
                              mirror=False, generator=gen)
        if ep.result == seat_a:
            wins += 1
    return {"wins": wins, "games": games}


def _pool_run(fn, total, workers, mk_args):
    per = total // workers
    rem = total - per * workers
    jobs = [mk_args(w, per + (1 if w < rem else 0)) for w in range(workers)
            if per + (1 if w < rem else 0) > 0]
    if len(jobs) == 1:
        stats = [fn(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            stats = pool.map(fn, jobs)
    return (sum(s["wins"] for s in stats), sum(s["games"] for s in stats))


def main():
    sys.path.insert(0, REPO)
    from ptcg.ei import wilson
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games-search", type=int, default=300)
    ap.add_argument("--games-raw", type=int, default=60)
    ap.add_argument("--anchor", default="champ/phase3-generalist-r120.pt")
    ap.add_argument("--anchor-games", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    def block(w, n):
        lo, hi = wilson(w, n)
        return {"wins": w, "games": n, "wr": w / max(n, 1),
                "lo": lo, "hi": hi}

    w, n = _pool_run(_search_chunk, a.games_search, a.workers,
                     lambda i, g: (g, a.seed + 100 * i, a.candidate,
                                   a.incumbent, a.k, a.sims))
    search_h2h = block(w, n)
    w, n = _pool_run(_raw_chunk, a.games_raw, a.workers,
                     lambda i, g: (g, a.seed + 1000 + 100 * i, a.candidate,
                                   a.incumbent))
    raw_h2h = block(w, n)
    w, n = _pool_run(_raw_chunk, a.games_raw, a.workers,
                     lambda i, g: (g, a.seed + 2000 + 100 * i, a.candidate,
                                   "random"))
    vs_random = block(w, n)
    anchor = {}
    if a.anchor and os.path.exists(a.anchor):
        w, n = _pool_run(_search_chunk, a.anchor_games, a.workers,
                         lambda i, g: (g, a.seed + 3000 + 100 * i,
                                       a.candidate, a.anchor, a.k, a.sims))
        anchor = block(w, n)
    promote = (search_h2h["lo"] > 0.50
               and vs_random["wr"] >= 0.85
               and (not anchor or anchor["wr"] >= 0.55))
    out = {"search_h2h": search_h2h, "raw_h2h": raw_h2h,
           "vs_random": vs_random, "vs_anchor_search": anchor,
           "promote": promote, "args": vars(a)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("gate_ei:", "PROMOTE" if promote else "REJECT",
          json.dumps({k: v for k, v in out.items() if k != "args"}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the calibration audit script**

Create `scripts/audit_value_phase.py`:

```python
"""Non-gating: the turn-boundary value-jump statistic (phase-4 diagnosis).
For each turn transition in random-play games, consistency demands
v(first select of new turn, acting seat) ~= -v(last select of prev turn,
prev seat); the jump is |v_new + v_prev|. Expert iteration should shrink
the mean jump across cycles.
Usage: python scripts/audit_value_phase.py --ckpt <path> --out <json> [--games 6]"""
import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    import numpy as np
    import torch

    from ptcg.cards import build_tables
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.featurize import featurize_state
    from ptcg.model import collate_states
    from ptcg.tracker import BeliefTracker
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gate_ei import _load_policy
    tables = build_tables()
    net = _load_policy(a.ckpt, tables)
    deck = load_sample_deck()
    jumps, all_v = [], []
    for g in range(a.games):
        s = BattleSession(deck, deck)
        try:
            rng = random.Random(a.seed + g)
            trk = {0: BeliefTracker(0), 1: BeliefTracker(1)}
            prev_v = prev_turn = None
            for _ in range(120):
                if s.done:
                    break
                seat = s.obs["current"]["yourIndex"]
                turn = s.obs["current"]["turn"]
                trk[seat].update(s.obs.get("logs") or [])
                ts = featurize_state(s.obs, seat, deck, trk[seat].snapshot(),
                                     tables)
                with torch.no_grad():
                    v = float(net.public_value(net.encode(
                        collate_states([ts]))))
                all_v.append(v)
                if prev_v is not None and prev_turn is not None \
                        and turn != prev_turn:
                    jumps.append(abs(v + prev_v))
                prev_v, prev_turn = v, turn
                s.select(random_picks(s.obs, rng))
        finally:
            s.close()
    out = {"mean_jump": float(np.mean(jumps)) if jumps else None,
           "n_jumps": len(jumps), "mean_abs_v": float(np.mean(np.abs(all_v))),
           "games": a.games}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("audit_value_phase:", json.dumps(out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke both (tiny)**

Run: `PYTHONIOENCODING=utf-8 python scripts/gate_ei.py --candidate runs/ei/dev-smoke/ckpt.pt --incumbent submission_src/policy.pt --out runs/ei/dev-smoke/gate.json --games-search 6 --games-raw 4 --anchor-games 4 --k 1 --sims 6 --workers 2 --seed 3`
Expected: completes in ~5-15 min, prints `gate_ei: PROMOTE|REJECT {...}` (verdict at these ns is noise — this smoke proves the harness), writes `runs/ei/dev-smoke/gate.json` with all four blocks + `promote`.

Run: `PYTHONIOENCODING=utf-8 python scripts/audit_value_phase.py --ckpt submission_src/policy.pt --out runs/ei/dev-smoke/audit.json --games 3`
Expected: prints `audit_value_phase: {"mean_jump": <float>, ...}`; json written.

- [ ] **Step 4: Commit**

```bash
git add scripts/gate_ei.py scripts/audit_value_phase.py
git commit -m "add ci promotion gate and value-phase calibration audit"
```

---

### Task 8: `scripts/ei_loop.py` — idempotent cycle driver

**Files:**
- Create: `scripts/ei_loop.py`
- Test: `tests/test_ei_loop.py`

**Interfaces:**
- Consumes: Tasks 3/6/7 CLIs (invoked via `subprocess.run([sys.executable, ...])`, one engine process at a time); their completion markers (`manifest.json`, `ckpt.pt`, `gate.json`).
- Produces: CLI `python scripts/ei_loop.py --run-id <id> --cycles N --start-ckpt <path> --games-per-cycle G --k K --sims S --gate-games GS --gate-raw GR --workers W [--kl-first-cycles 2] [--kl-coef 0.02] [--device cpu] [--seed 1]`. Layout: `runs/ei/<run-id>/state.json` (`{"cycle": int, "incumbent": str, "history": [...]}`) and `runs/ei/<run-id>/cycle-<n>/{data/, ckpt.pt, gate.json, audit.json}`. Stage functions `_stage_gen/_stage_train/_stage_gate` are skipped when their marker exists (idempotent resume); a REJECTed cycle keeps the incumbent and continues. Module-level `run_cycle(run_dir, n, incumbent, args, runner=subprocess.run) -> (new_incumbent, verdict)` for testability.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ei_loop.py` (pure logic — fake runner, no engine):

```python
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import ei_loop


def _fake_runner_factory(promote, calls):
    def fake_run(cmd, **kw):
        calls.append(cmd)
        # emulate each stage's completion marker
        joined = " ".join(str(c) for c in cmd)
        if "gen_ei.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump({"games": 1, "moves": 1}, f)
        elif "train_ei.py" in joined:
            out = cmd[cmd.index("--ckpt-out") + 1]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write("ckpt")
        elif "gate_ei.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            with open(out, "w") as f:
                json.dump({"promote": promote}, f)
        elif "audit_value_phase.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            with open(out, "w") as f:
                json.dump({"mean_jump": 0.1}, f)

        class R:
            returncode = 0
        return R()
    return fake_run


def _args(tmp):
    return ei_loop.LoopArgs(
        run_dir=str(tmp), start_ckpt="start.pt", games_per_cycle=1,
        k=1, sims=4, gate_games=4, gate_raw=2, workers=1,
        kl_first_cycles=1, kl_coef=0.02, device="cpu", seed=1)


def test_promote_advances_incumbent_and_resume_skips(tmp_path):
    calls = []
    a = _args(tmp_path)
    inc, verdict = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                     runner=_fake_runner_factory(True, calls))
    assert verdict is True
    assert inc == os.path.join(str(tmp_path), "cycle-0", "ckpt.pt")
    n_first = len(calls)
    assert n_first >= 3
    # resume: markers exist -> zero subprocess calls, same outcome
    calls2 = []
    inc2, verdict2 = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                       runner=_fake_runner_factory(True, calls2))
    assert (inc2, verdict2) == (inc, True)
    assert calls2 == []


def test_reject_keeps_incumbent(tmp_path):
    calls = []
    a = _args(tmp_path)
    inc, verdict = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                     runner=_fake_runner_factory(False, calls))
    assert verdict is False
    assert inc == "start.pt"


def test_state_roundtrip(tmp_path):
    p = os.path.join(str(tmp_path), "state.json")
    st = {"cycle": 2, "incumbent": "x.pt", "history": [{"cycle": 0}]}
    ei_loop.save_state(p, st)
    assert ei_loop.load_state(p) == st
    assert ei_loop.load_state(os.path.join(str(tmp_path), "nope.json")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_ei_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ei_loop'`

- [ ] **Step 3: Implement**

Create `scripts/ei_loop.py`:

```python
"""Resumable expert-iteration driver: generate -> train -> gate -> promote.
Each stage is a subprocess (one engine process at a time) and is SKIPPED
when its completion marker exists, so re-running the same command resumes
from the interrupted stage. A REJECTed cycle keeps the incumbent and the
loop continues with fresh data next cycle.
Usage:
  python scripts/ei_loop.py --run-id ei-a --cycles 3 \
      --start-ckpt submission_src/policy.pt --games-per-cycle 200 \
      --k 3 --sims 32 --gate-games 300 --gate-raw 60 --workers 3
State: runs/ei/<run-id>/state.json ; artifacts under cycle-<n>/."""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


@dataclass
class LoopArgs:
    run_dir: str
    start_ckpt: str
    games_per_cycle: int
    k: int
    sims: int
    gate_games: int
    gate_raw: int
    workers: int
    kl_first_cycles: int = 2
    kl_coef: float = 0.02
    device: str = "cpu"
    seed: int = 1


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _run(runner, cmd):
    r = runner(cmd)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"stage failed: {' '.join(str(c) for c in cmd)}")


def run_cycle(run_dir, n, incumbent, a, runner=None):
    """Returns (new_incumbent, promoted_bool). Idempotent per stage."""
    runner = runner or (lambda cmd: subprocess.run(cmd))
    cdir = os.path.join(run_dir, f"cycle-{n}")
    data = os.path.join(cdir, "data")
    ckpt = os.path.join(cdir, "ckpt.pt")
    gate = os.path.join(cdir, "gate.json")
    audit = os.path.join(cdir, "audit.json")
    py = sys.executable
    if not os.path.exists(os.path.join(data, "manifest.json")):
        _run(runner, [py, os.path.join(SCRIPTS, "gen_ei.py"),
                      "--ckpt", incumbent, "--out", data,
                      "--games", str(a.games_per_cycle),
                      "--workers", str(a.workers), "--k", str(a.k),
                      "--sims", str(a.sims), "--seed", str(a.seed + n)])
    if not os.path.exists(ckpt):
        cmd = [py, os.path.join(SCRIPTS, "train_ei.py"),
               "--data", data, "--ckpt-in", incumbent, "--ckpt-out", ckpt,
               "--device", a.device, "--seed", str(a.seed + n)]
        prev = os.path.join(run_dir, f"cycle-{n - 1}", "data")
        if n > 0 and os.path.isdir(prev):
            cmd += ["--replay", prev]
        if n < a.kl_first_cycles:
            cmd += ["--kl-coef", str(a.kl_coef)]
        _run(runner, cmd)
    if not os.path.exists(gate):
        _run(runner, [py, os.path.join(SCRIPTS, "gate_ei.py"),
                      "--candidate", ckpt, "--incumbent", incumbent,
                      "--out", gate, "--games-search", str(a.gate_games),
                      "--games-raw", str(a.gate_raw), "--k", str(a.k),
                      "--sims", str(a.sims), "--workers", str(a.workers),
                      "--seed", str(a.seed + 10 * n)])
    if not os.path.exists(audit):
        _run(runner, [py, os.path.join(SCRIPTS, "audit_value_phase.py"),
                      "--ckpt", ckpt, "--out", audit])
    with open(gate) as f:
        promoted = bool(json.load(f).get("promote"))
    return (ckpt if promoted else incumbent), promoted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--start-ckpt", required=True)
    ap.add_argument("--games-per-cycle", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--gate-games", type=int, default=300)
    ap.add_argument("--gate-raw", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--kl-first-cycles", type=int, default=2)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ag = ap.parse_args()
    run_dir = os.path.join(REPO, "runs", "ei", ag.run_id)
    sp = os.path.join(run_dir, "state.json")
    st = load_state(sp) or {"cycle": 0, "incumbent": ag.start_ckpt,
                            "history": []}
    a = LoopArgs(run_dir, ag.start_ckpt, ag.games_per_cycle, ag.k, ag.sims,
                 ag.gate_games, ag.gate_raw, ag.workers, ag.kl_first_cycles,
                 ag.kl_coef, ag.device, ag.seed)
    while st["cycle"] < ag.cycles:
        n = st["cycle"]
        print(f"ei_loop: cycle {n} incumbent={st['incumbent']}", flush=True)
        inc, promoted = run_cycle(run_dir, n, st["incumbent"], a)
        st["history"].append({"cycle": n, "promoted": promoted,
                              "incumbent_after": inc})
        st["incumbent"] = inc
        st["cycle"] = n + 1
        save_state(sp, st)
    print("ei_loop: done", json.dumps(st))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ei_loop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/ei_loop.py tests/test_ei_loop.py
git commit -m "add resumable expert-iteration cycle driver"
```

---

### Task 9: cloud playbook + stage-0 miniature cycle (mechanism validation)

**Files:**
- Create: `scripts/cloud_setup.sh`
- Create: `docs/phase5-run.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the stage-0 verdict recorded in `.superpowers/sdd/progress.md`; the operator playbook for stage 1.

- [ ] **Step 1: Write `scripts/cloud_setup.sh`**

```bash
#!/usr/bin/env bash
# Rented-box setup for phase-5 expert iteration (Linux x86, Ubuntu-ish).
# Usage: bash scripts/cloud_setup.sh [cpu|cu121]
set -euo pipefail
FLAVOR="${1:-cpu}"
sudo apt-get update -y && sudo apt-get install -y python3.11 python3.11-venv rsync
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
if [ "$FLAVOR" = "cu121" ]; then
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi
pip install numpy pytest
export PYTHONIOENCODING=utf-8
# engine: the repo's default PTCG_ENGINE_DIR resolution works from repo root
python -c "import sys; sys.path.insert(0,'.'); from ptcg.cards import build_tables; t=build_tables(); print('engine ok, rows', t.n_rows)"
echo "setup ok ($FLAVOR)"
```

- [ ] **Step 2: Write `docs/phase5-run.md`** (operator playbook)

```markdown
# Phase-5 expert iteration — run playbook

## Stage 1 box
- 64-128 vCPU + 1x 4090/A100-class GPU, Linux x86, >=200 GB disk.
- `git clone` the repo (or rsync the working tree), then
  `bash scripts/cloud_setup.sh cu121` (learner) — generation is CPU-only.

## Launch (detached; never as a child of an agent session)
    nohup .venv/bin/python scripts/ei_loop.py --run-id ei-a --cycles 5 \
      --start-ckpt submission_src/policy.pt --games-per-cycle 20000 \
      --k 3 --sims 32 --gate-games 300 --gate-raw 60 \
      --workers 60 --device cuda --seed 1 > runs/ei-a.log 2>&1 &

## Resume after any interruption
Re-run the SAME command: completed stages are skipped via their markers
(manifest.json / ckpt.pt / gate.json), the loop continues mid-cycle.

## Artifact sync (run from the LOCAL machine, repeat while the run lives)
    rsync -avz cloud:~/Pokemon-Kaggle/runs/ei/ei-a/ runs/ei/ei-a/ \
      --exclude 'data/worker-*'          # gate/ckpt/state only; add data if wanted

## Costs (measured basis, d224)
- generation ~1-1.5 s/searched move/core -> 64 cores ~3-5k games/hr
- one 20-40k-game cycle: ~$15-25 gen + a few $ train/gate
- stage-1 budget envelope ~$100-500; stage-2 trigger per the spec

## Ship checklist (mid-month + end)
1. promoted ckpt -> `python scripts/extract_policy.py` equivalent: save the
   `"policy"` state dict as submission_src/policy.pt (backup the old one)
2. `python scripts/make_submission.py` and `--no-search` (both must pass)
3. `python scripts/test_submission.py 4 --small-search` (legal, searched>0)
4. `python scripts/gate_ei.py --candidate <new> --incumbent <old policy.pt>`
   — ship only on PROMOTE
5. user uploads dist/submission.zip; rollback = previous zip pair

## Stage triggers (pre-registered, from the spec)
- stage-2 (capacity) iff cycles still PROMOTE but per-cycle search-wrapped
  gains fall below ~+2%: d384/d512 init-by-distillation, distill back to a
  CPU-fast student before shipping.
- abort/diagnose iff two consecutive REJECTs: check entropy, audit.json
  mean_jump trend, replay ratio — one variable at a time.
```

- [ ] **Step 3: Run stage 0 locally (the mechanism gate)**

This is a ~1-2 h local run; launch it detached or let the controller run it in the background (never inside a subagent):

Run: `PYTHONIOENCODING=utf-8 python scripts/ei_loop.py --run-id ei-stage0 --cycles 1 --start-ckpt submission_src/policy.pt --games-per-cycle 80 --k 2 --sims 16 --gate-games 40 --gate-raw 20 --workers 3 --seed 1`
Expected: completes; `runs/ei/ei-stage0/cycle-0/` contains `data/manifest.json`, `ckpt.pt`, `gate.json`, `audit.json`; `state.json` shows cycle 1 with a history entry. The gate verdict at n=40 is noise — stage-0 SUCCESS is: losses in train output finite and `loss_pi` < its first-minibatch value, all artifacts present, no crashes.

- [ ] **Step 4: Verify resume idempotency**

Re-run the exact same command.
Expected: completes in seconds — all stages skipped (markers exist), state unchanged (`cycle: 1`).

- [ ] **Step 5: Record stage-0 results and commit**

Append the stage-0 verdict (games, moves, losses, gate numbers, audit mean_jump, resume check) to `.superpowers/sdd/progress.md` under the phase-5 header.

```bash
git add scripts/cloud_setup.sh docs/phase5-run.md .superpowers/sdd/progress.md
git commit -m "add phase-5 cloud playbook and record stage-0 mechanism validation"
```

---

## Plan Self-Review (performed while writing)

- **Spec coverage:** recorder with sims budget + raw visit counts (Task 2 = spec §Generate; temperature stays train-time per spec via raw `visits`), trainer losses incl. value-on-every-state, aux upkeep, replay mixing, KL warmup, temp knob (Tasks 4-6 = §Train), gate with pre-registered promotion rule + anchors + calibration audit (Task 7 = §Gate; exact thresholds lo>0.50, vs-random ≥0.85, anchor ≥0.55 now pinned), idempotent driver (Task 8 = §Promote & loop), cloud setup + playbook + stage-0 (Task 9 = §Infrastructure/§Stages). MoveStats exposure (Task 1) is the enabling additive change with a behavior-identity guard. Out-of-scope items (PSRO, inference server, packaging changes) have no tasks — matching the spec.
- **Type consistency:** `EIStep(player, state, esel, priv_state, actions, visits)` consumed by Tasks 4/5 via those exact names (`aux_targets` field contract asserted in Task 2's test); `wilson` lives in `ptcg/ei.py` and is imported by `gate_ei.py`; `load_policy`/`_load_policy` semantics identical everywhere (bare state dict OR `{"policy": ...}`); `run_cycle(run_dir, n, incumbent, args, runner)` matches the Task-8 tests; driver stage CLIs match Tasks 3/6/7 flags exactly (`--out`, `--ckpt-out`, `--games-search`).
- **Placeholder scan:** every code step carries complete, runnable code; no TBD/TODO/"similar to Task N".
- **Arena reuse:** one SearchSession per worker in gen_ei and gate_ei (arenas have no native free; per-game sessions would leak across long chunks) — threaded via play_search_game(session=...).
