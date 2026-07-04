# PTCG training pipeline (phase 2) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A synchronous round-based self-play PPO loop (CPU actors, GTX 1060
learner) that demonstrably learns on the sample deck, per
`docs/superpowers/specs/2026-07-04-training-pipeline-phase2-design.md`.

**Architecture:** Rounds of mirror self-play collected by 3 actor processes
(B==1 acting path from phase 1), trajectories stored as featurized tensors and
consumed once by a GPU learner running batched-replay PPO with the privileged
critic; checkpoint/resume at round boundaries; eval vs random and past
checkpoints every 5 rounds.

**Tech Stack:** Python 3.12 training venv with a Pascal-capable torch CUDA
build (spiked in Task 0); phase-1 `ptcg` package unchanged except one opt-in
rollout parameter; numpy, pytest, matplotlib.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-training-pipeline-phase2-design.md`.
  On conflict the spec wins; deviations must be listed in the final report.
  Phase-1 contracts remain binding: shared pick loop, learner replays stored
  pick order and never re-samples; dropout stays 0.0; the option list is the
  only legality oracle; featurizers are pure functions; never modify anything
  under `pokemon-tcg-ai-battle/`; one battle per process; engine-dependent
  tests assert invariants and ranges, never exact outcomes.
- Ratio-contract refinement (spec §Learner): bit-exact within a
  (device, dtype, batching) configuration, tested vs the B==1 path at atol=0 on
  CPU; across configurations, epoch-0 `max|ratio − 1| < 1e-3` asserted every
  round, abort on violation.
- Learner trains fp32 (no mixed precision on Pascal). `device` is config;
  documented fallback is `cpu`.
- Phase-1 modules are consumed, not modified — sole exception: `play_game`
  gains an opt-in `priv_viz` parameter (Task 3).
- All tests run CPU with `tiny_config`; a full pytest invocation is one
  process; never run pytest in parallel.
- Existing 3.14 install stays the phase-1 test environment; the training venv
  is `venv-train/` (git-ignored) created in Task 0.
- Commit style: short lowercase messages, commit directly on `main`.
- Windows: `python` (never `python3`); multiprocessing uses spawn; every
  process entry function is a top-level module function;
  `if __name__ == "__main__"` guards on scripts.

## File Structure

```
venv-train/                 — Python 3.12 + torch-cu venv (git-ignored, Task 0)
benchmarks/spike_gpu.py     — Pascal capability probe (Task 0)
benchmarks/RESULTS-gpu.md   — spike results (Task 0)
ptcg/replay.py              — batched deterministic replay (Task 1)
ptcg/ppo.py                 — GAE, PPO losses, aux targets (Task 2)
ptcg/rollout.py             — modify: play_game(priv_viz=False) (Task 3)
ptcg/trainloop.py           — TrainConfig, run dirs, serialization, checkpoints
                              (Task 4); learner_update (Task 6); train() (Task 7)
ptcg/actors.py              — actor process, pool, play_versus, eval_round (Tasks 5, 8)
scripts/train.py            — CLI entry (Task 9)
scripts/plot_run.py         — metrics plots (Task 9)
tests/test_replay.py … tests/test_trainloop.py — per task, named below
benchmarks/RESULTS-train-smoke.md — smoke-run record (Task 9)
```

---

### Task 0: Training venv + GTX 1060 capability spike

**Files:**
- Create: `benchmarks/spike_gpu.py`, `benchmarks/RESULTS-gpu.md`
- Modify: `.gitignore` (add `venv-train/`, `runs/`)

**Interfaces:**
- Produces: a working `venv-train\Scripts\python` with numpy, pytest,
  matplotlib, an sm_61-capable torch, and `ptcg` installed editable — or a
  recorded CPU fallback decision. Later tasks run training commands with this
  interpreter and tests with the base interpreter.

- [ ] **Step 1: Find or install Python 3.12**

Run: `py -0p`
If no 3.11/3.12/3.13 listed: `winget install -e --id Python.Python.3.12`, then
re-run `py -0p`. If winget fails (store policy, no admin), STOP and report
BLOCKED — the human must install Python 3.12.

- [ ] **Step 2: Create the venv and install non-torch deps**

```
py -3.12 -m venv venv-train
venv-train\Scripts\python -m pip install --upgrade pip
venv-train\Scripts\python -m pip install numpy pytest matplotlib
```

- [ ] **Step 3: Install a Pascal-capable torch (decision ladder)**

Try in order; after each install run the arch check below and stop at the
first success:

```
venv-train\Scripts\python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
venv-train\Scripts\python -m pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
venv-train\Scripts\python -m pip install torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Arch check (success = prints `OK`):

```
venv-train\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'no cuda'; assert 'sm_61' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list(); assert torch.cuda.get_device_capability(0) == (6, 1), torch.cuda.get_device_capability(0); print('OK')"
```

If all three fail: record CPU fallback in RESULTS-gpu.md (`device: cpu`), keep
the cu-less base torch in the venv via
`venv-train\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`,
and continue the plan — everything downstream takes `device` from config.

- [ ] **Step 4: Write and run the spike**

`benchmarks/spike_gpu.py`:

```python
"""GTX 1060 spike: matmul+backward on cuda, encoder fwd/bwd timing, suite compat."""
import time

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev,
      "| arch", torch.cuda.get_arch_list() if dev == "cuda" else "-")

x = torch.randn(512, 512, device=dev, requires_grad=True)
(x @ x).sum().backward()
assert torch.isfinite(x.grad).all()
print("matmul+backward ok")

from ptcg.cards import build_tables
from ptcg.model import Encoder, student_config

enc = Encoder(student_config(build_tables())).to(dev)
batch = {
    "card": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "numeric": torch.randn(256, 192, 40, device=dev),
    "owner": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "zone": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "kind": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "pos": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "mask": torch.ones(256, 192, dtype=torch.bool, device=dev),
}
for _ in range(2):  # warmup
    enc(batch).sum().backward()
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    enc(batch).sum().backward()
if dev == "cuda":
    torch.cuda.synchronize()
print(f"student encoder fwd+bwd b256: {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")
```

Run: `venv-train\Scripts\python -m pip install -e . --no-deps && venv-train\Scripts\python benchmarks/spike_gpu.py`
(`--no-deps` because pyproject pins `torch>=2.4`; the venv's torch was chosen
by the Pascal ladder and must not be replaced — the 2.3.1 fallback rung would
otherwise conflict.)
Expected: `matmul+backward ok` and a timing line.

- [ ] **Step 5: Run the phase-1 suite on the venv torch**

Run: `venv-train\Scripts\python -m pytest tests/ -q`
Expected: 37 passed (CPU; proves the phase-1 code runs on the older torch).
Any failure here is a compat bug — STOP and report it with the output.

- [ ] **Step 6: Record and commit**

Write `benchmarks/RESULTS-gpu.md`: date, chosen torch build, arch list, the
two spike output lines, suite result, and the decided `device` value.
Append `venv-train/` and `runs/` to `.gitignore`.

```bash
git add benchmarks/spike_gpu.py benchmarks/RESULTS-gpu.md .gitignore
git commit -m "gpu training venv spike"
```

---

### Task 1: Batched deterministic replay

**Files:**
- Create: `ptcg/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `ptcg.model.collate_states/collate_selects/PolicyModel.encode/option_logits`;
  `ptcg.action.replay_logprob`, `run_pick_loop` (reference implementations).
- Produces:

```python
def batched_replay(model, trunk, sb, selb, picks_list) -> tuple[Tensor, Tensor]
    # core: caller supplies the trunk encode; returns ([B] logp, [B] entropy),
    # both differentiable. Shares option_logits with the B==1 path.
def replay_logprob_batched(model, states, sels, picks_list, device=None)
    -> tuple[Tensor, Tensor]   # convenience: collates, moves to device, encodes
```

- [ ] **Step 1: Write the failing test**

`tests/test_replay.py`:

```python
import torch
from ptcg.action import replay_logprob, run_pick_loop
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, collate_selects, collate_states, tiny_config
from ptcg.replay import replay_logprob_batched
from ptcg.rollout import play_game


def _collect_steps(n=24):
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(3)
    steps = []
    while len(steps) < n:
        ep = play_game(m, (deck, list(deck)), tables, generator=g)
        steps.extend(ep.steps)
    return tables, m, steps[:n]


def test_batched_matches_b1_exactly():
    tables, m, steps = _collect_steps()
    states = [s.state for s in steps]
    sels = [s.esel for s in steps]
    picks = [s.picks for s in steps]
    lp1 = replay_logprob(m, states, sels, picks)
    lpb, entb = replay_logprob_batched(m, states, sels, picks)
    assert torch.allclose(lpb, lp1, atol=0, rtol=0), (lpb - lp1).abs().max()
    # entropy reference via the B==1 forced loop
    ents = []
    for s in steps:
        sb = collate_states([s.state])
        selb = collate_selects([s.esel])
        trunk = m.encode(sb)
        _, _, ent = run_pick_loop(m, trunk, sb, selb, forced=list(s.picks))
        ents.append(ent)
    assert torch.allclose(entb, torch.stack(ents), atol=0, rtol=0)


def test_batched_replay_backward():
    tables, m, steps = _collect_steps(8)
    lpb, entb = replay_logprob_batched(
        m, [s.state for s in steps], [s.esel for s in steps],
        [s.picks for s in steps])
    (lpb.sum() + entb.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in m.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.replay'`

- [ ] **Step 3: Implement `ptcg/replay.py`**

```python
import torch
from torch.distributions import Categorical

from .model import collate_selects, collate_states


def batched_replay(model, trunk, sb, selb, picks_list):
    """Deterministic lockstep replay of stored pick sequences over a shared trunk.

    Mirrors action.run_pick_loop(forced=...) semantics per row: force each
    stored pick in order; force one done step only where the stored sequence
    ended before max_count; never re-sample. picked is cloned before each
    mutation (same autograd hazard as the B==1 loop).
    """
    B, O = selb["opt_type"].shape
    dev = trunk.device
    max_count = selb["max_count_t"].to(dev)
    n_picks = torch.tensor([len(p) for p in picks_list], device=dev)
    depth = int(n_picks.max().item()) if B else 0
    picked = torch.zeros((B, O + 1), dtype=torch.bool, device=dev)
    logp = trunk.new_zeros(B)
    ent = trunk.new_zeros(B)
    for step in range(depth + 1):
        takes_pick = n_picks > step
        forces_done = (n_picks == step) & (n_picks < max_count)
        alive = takes_pick | forces_done
        if not bool(alive.any()):
            break
        actions = torch.full((B,), O, dtype=torch.int64, device=dev)
        for i, picks in enumerate(picks_list):
            if step < len(picks):
                actions[i] = picks[step]
        logits = model.option_logits(trunk, sb, selb, picked)
        dist = Categorical(logits=logits)
        lp = dist.log_prob(actions)
        en = dist.entropy()
        zero = torch.zeros((), device=dev, dtype=lp.dtype)
        logp = logp + torch.where(alive, lp, zero)
        ent = ent + torch.where(alive, en, zero)
        new_picked = picked.clone()
        rows = takes_pick.nonzero(as_tuple=True)[0]
        new_picked[rows, actions[rows]] = True
        picked = new_picked
    return logp, ent


def replay_logprob_batched(model, states, sels, picks_list, device=None):
    sb = collate_states(states)
    selb = collate_selects(sels)
    if device is not None:
        sb = {k: v.to(device) for k, v in sb.items()}
        selb = {k: v.to(device) for k, v in selb.items()}
    trunk = model.encode(sb)
    return batched_replay(model, trunk, sb, selb, picks_list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_replay.py -v`
Expected: 2 PASS.
Decision rule if `test_batched_matches_b1_exactly` fails ONLY in the last
float bits (max abs diff ≤ 1e-6, no structural mismatch): CPU GEMM kernels for
batched vs single-row shapes may differ; relax that one assertion to
`atol=1e-6, rtol=0`, record the deviation in your report, and keep the
backward test at exact semantics. Any larger or structural difference is a
bug in the lockstep logic — fix it, do not relax.

- [ ] **Step 5: Commit**

```bash
git add ptcg/replay.py tests/test_replay.py
git commit -m "batched deterministic replay"
```

---

### Task 2: GAE and PPO losses

**Files:**
- Create: `ptcg/ppo.py`
- Test: `tests/test_ppo_units.py`

**Interfaces:**
- Consumes: `ptcg.model.collate_states`, `CriticModel`; `Episode`/`Step`
  (`ptcg.rollout`); featurize constants (`OWNER_SELF/OWNER_OPP`, zone HAND=2,
  `KIND_ENTITY`, `F_PRIZEN`, rows 1–2 = player summaries).
- Produces:

```python
def compute_gae(values: list[float], terminal_reward: float,
                lam=0.95, gamma=1.0) -> tuple[list[float], list[float]]
def assemble_advantages(episodes, critic, device=None, lam=0.95, gamma=1.0,
                        normalize=True)
    -> tuple[list[Step], Tensor, Tensor, Tensor]   # steps, old_lp, adv, ret
def ppo_policy_loss(new_lp, old_lp, adv, clip=0.2)
    -> tuple[Tensor, Tensor, Tensor]               # pg_loss, ratio, approx_kl
def aux_targets(steps, tables, opp_deck: list[int])
    -> tuple[Tensor, Tensor, Tensor]  # prize_diff [B], decklist [B, n_rows], hand [B, n_rows]
```

- [ ] **Step 1: Write the failing test**

`tests/test_ppo_units.py`:

```python
import math

import torch
from ptcg.ppo import compute_gae, ppo_policy_loss


def test_gae_hand_computed():
    # values [0.5, 0.0], terminal reward 1.0, lam=0.5, gamma=1.0
    # delta1 = 0.0 - 0.5 = -0.5 ; delta2 = 1.0 - 0.0 = 1.0
    # adv2 = 1.0 ; adv1 = -0.5 + 0.5*1.0 = 0.0
    adv, ret = compute_gae([0.5, 0.0], 1.0, lam=0.5, gamma=1.0)
    assert math.isclose(adv[0], 0.0, abs_tol=1e-9)
    assert math.isclose(adv[1], 1.0, abs_tol=1e-9)
    assert math.isclose(ret[0], 0.5, abs_tol=1e-9)
    assert math.isclose(ret[1], 1.0, abs_tol=1e-9)


def test_gae_matches_smoke_test_reference():
    # the phase-1 smoke test's _gae over deltas must agree
    def _gae(deltas, gamma=1.0, lam=0.95):
        adv, out = 0.0, []
        for d in reversed(deltas):
            adv = d + gamma * lam * adv
            out.append(adv)
        return list(reversed(out))

    vals, rw = [0.2, -0.1, 0.4], -1.0
    deltas = [(vals[j + 1] if j + 1 < len(vals) else rw) - vals[j]
              for j in range(len(vals))]
    ref = _gae(deltas)
    adv, _ = compute_gae(vals, rw)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(adv, ref))


def test_ppo_policy_loss_clip_and_kl():
    old = torch.tensor([0.0, 0.0])
    new = torch.tensor([math.log(2.0), math.log(0.25)])
    adv = torch.tensor([1.0, 1.0])
    pg, ratio, kl = ppo_policy_loss(new, old, adv, clip=0.2)
    # ratios 2.0 and 0.25; positive adv -> min(r, clip(r)) = (1.2, 0.25)
    assert torch.allclose(ratio, torch.tensor([2.0, 0.25]))
    assert math.isclose(float(pg), -(1.2 + 0.25) / 2, abs_tol=1e-6)
    assert math.isclose(float(kl), float((old - new).mean()), abs_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ppo_units.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.ppo'`

- [ ] **Step 3: Implement `ptcg/ppo.py`**

```python
import numpy as np
import torch

from .cards import card_row
from .featurize import F_PRIZEN, KIND_ENTITY, OWNER_OPP, OWNER_SELF
from .model import collate_states

AREA_HAND = 2
_PSUM_SELF, _PSUM_OPP = 1, 2  # fixed special-token rows


def compute_gae(values, terminal_reward, lam=0.95, gamma=1.0):
    """Per-seat GAE with terminal-only reward; matches the phase-1 smoke test."""
    adv_acc, advs = 0.0, []
    for j in reversed(range(len(values))):
        nxt = values[j + 1] if j + 1 < len(values) else terminal_reward
        delta = gamma * nxt - values[j]
        adv_acc = delta + gamma * lam * adv_acc
        advs.append(adv_acc)
    advs.reverse()
    rets = [a + v for a, v in zip(advs, values)]
    return advs, rets


def assemble_advantages(episodes, critic, device=None, lam=0.95, gamma=1.0,
                        normalize=True):
    steps = [s for ep in episodes for s in ep.steps]
    with torch.no_grad():
        chunks = []
        for lo in range(0, len(steps), 256):
            batch = collate_states([s.priv_state for s in steps[lo:lo + 256]])
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            chunks.append(critic(batch).cpu())
        pv = torch.cat(chunks) if chunks else torch.zeros(0, 2)
    adv = torch.zeros(len(steps))
    ret = torch.zeros(len(steps))
    off = 0
    for ep in episodes:
        idx = list(range(off, off + len(ep.steps)))
        off += len(ep.steps)
        for seat in (0, 1):
            rows = [i for i in idx if steps[i].player == seat]
            vals = [float(pv[i, seat]) for i in rows]
            a, r = compute_gae(vals, ep.rewards[seat], lam, gamma)
            for k, i in enumerate(rows):
                adv[i] = a[k]
                ret[i] = r[k]
    if normalize and len(steps) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    old_lp = torch.tensor([s.logprob for s in steps])
    return steps, old_lp, adv, ret


def ppo_policy_loss(new_lp, old_lp, adv, clip=0.2):
    ratio = (new_lp - old_lp).exp()
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    pg = -torch.min(ratio * adv, clipped * adv).mean()
    approx_kl = (old_lp - new_lp).mean()
    return pg, ratio, approx_kl


def aux_targets(steps, tables, opp_deck):
    """Targets for the train-only aux heads, from stored tensors only.

    prize_diff: (opp prizes remaining - own prizes remaining) read from the
    public state's player-summary rows (positive = acting seat is ahead).
    decklist: full opponent decklist as card-table-row counts (constant under
    mirror play; kept per-step for phase-3 portability).
    hand: true opponent hand counts from the privileged state. The privileged
    view is seat-0-fixed, so the acting seat's opponent is OWNER_OPP when
    player==0 and OWNER_SELF when player==1.
    """
    n_rows = tables.n_rows
    B = len(steps)
    pd = torch.zeros(B)
    dl = torch.zeros(B, n_rows)
    hd = torch.zeros(B, n_rows)
    dl_vec = torch.zeros(n_rows)
    for cid in opp_deck:
        dl_vec[card_row(cid, n_rows)] += 1.0
    for i, s in enumerate(steps):
        num = s.state.numeric
        pd[i] = float(num[_PSUM_OPP, F_PRIZEN] - num[_PSUM_SELF, F_PRIZEN]) * 6.0
        dl[i] = dl_vec
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

Run: `python -m pytest tests/test_ppo_units.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/ppo.py tests/test_ppo_units.py
git commit -m "gae and ppo losses with aux targets"
```

---

### Task 3: Rollout viz pass-through for the critic

**Files:**
- Modify: `ptcg/rollout.py` (play_game signature + one call site)
- Test: `tests/test_rollout_viz.py`

**Interfaces:**
- Produces: `play_game(model, decks, tables, generator=None, step_cap=5000,
  priv_viz=False, obs_log=None)` — when `priv_viz=True`, every step passes
  `viz=s.viz_current()` into `featurize_privileged` so privileged states carry
  true deck-order (and revealed-prize) tokens; when `obs_log` is a list, the
  raw obs dict of every step is appended to it (the spec's debug-sample hook,
  consumed by Task 5). Defaults preserve phase-1 behavior byte-for-byte.

- [ ] **Step 1: Write the failing test**

`tests/test_rollout_viz.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.featurize import KIND_ENTITY, OWNER_OPP
from ptcg.model import PolicyModel, tiny_config
from ptcg.rollout import play_game

AREA_DECK_Z = 1  # featurize does not export AreaType ids; literal as in phase-1 tests


def _opp_deck_entities(ts):
    return [i for i in range(ts.n)
            if ts.zone[i] == AREA_DECK_Z and ts.owner[i] == OWNER_OPP
            and ts.kind[i] == KIND_ENTITY]


def test_priv_viz_adds_deck_entities_and_default_unchanged():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(4)
    ep_off = play_game(m, (deck, list(deck)), tables, generator=g)
    assert all(_opp_deck_entities(s.priv_state) == [] for s in ep_off.steps)
    ep_on = play_game(m, (deck, list(deck)), tables, generator=g,
                      priv_viz=True)
    assert any(_opp_deck_entities(s.priv_state) for s in ep_on.steps)
    for s in ep_on.steps:
        assert s.priv_state.n == int(s.priv_state.mask.sum())


def test_obs_log_captures_raw_observations():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(5)
    log = []
    ep = play_game(m, (deck, list(deck)), tables, generator=g, obs_log=log)
    assert len(log) == len(ep.steps)
    assert all(isinstance(o, dict) and "current" in o for o in log)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rollout_viz.py -v`
Expected: FAIL with `TypeError: play_game() got an unexpected keyword argument 'priv_viz'`

- [ ] **Step 3: Implement**

In `ptcg/rollout.py`, change the signature to
`def play_game(model, decks, tables, generator=None, step_cap=5000, priv_viz=False, obs_log=None):`,
add immediately after `me = s.select_player`:

```python
            if obs_log is not None:
                obs_log.append(s.obs)
```

and replace the viz_hands/privileged-featurize block — reusing one
`viz_current()` call for both consumers:

```python
            vcur = s.viz_current() if (priv_viz or not (seen[0] and seen[1])) else None
            viz_hands = None
            if not (seen[0] and seen[1]) and vcur is not None:
                vp = vcur.get("players") or []
                if len(vp) == 2:
                    viz_hands = [vp[0].get("hand"), vp[1].get("hand")]
            pv = featurize_privileged(last_obs[0], last_obs[1], decks, tables,
                                      viz=vcur if priv_viz else None,
                                      viz_hands=viz_hands)
```

- [ ] **Step 4: Run the new test, then the full suite**

Run: `python -m pytest tests/test_rollout_viz.py -v` → 1 PASS
Run: `python -m pytest tests/ -q` → all pass (37 + new).

- [ ] **Step 5: Commit**

```bash
git add ptcg/rollout.py tests/test_rollout_viz.py
git commit -m "opt-in viz deck order for privileged rollout states"
```

---

### Task 4: TrainConfig, run directories, trajectory and checkpoint IO

**Files:**
- Create: `ptcg/trainloop.py`
- Test: `tests/test_trainloop.py`

**Interfaces:**
- Consumes: `Episode` (`ptcg.rollout`), `ModelConfig` constructors
  (`tiny_config/student_config/teacher_config`), `PolicyModel`, `CriticModel`,
  `critic_config` (`ptcg.model`), `FEATURIZER_VERSION` (`ptcg.featurize`).
- Produces (consumed by Tasks 5–9):

```python
@dataclass
class TrainConfig:
    run_dir: str; model_size: str = "student"      # tiny|student|teacher
    games_per_round: int = 192; actors: int = 3
    epochs: int = 2; minibatch: int = 512; lr: float = 3e-4
    clip: float = 0.2; lam: float = 0.95; gamma: float = 1.0
    ent_coef: float = 0.01; vf_coef: float = 0.5; critic_coef: float = 0.5
    aux_coef: float = 0.1; kl_stop: float = 0.02; grad_clip: float = 1.0
    ratio_gate: float = 1e-3; eval_every: int = 5
    eval_games_random: int = 200; eval_games_ckpt: int = 100
    device: str = "cpu"; seed: int = 0; step_cap: int = 5000

def model_config_for(size: str, tables) -> ModelConfig
def round_dir(cfg, n: int) -> Path            # <run_dir>/rounds/<n>
def checkpoint_path(cfg, n: int) -> Path      # <run_dir>/checkpoint-%04d.pt
def save_game(path, episode) -> None          # torch.save, .pt.tmp then rename
def load_round(cfg, n: int) -> list[Episode]
def save_checkpoint(cfg, n, policy, critic, optim) -> None   # atomic, CPU tensors
def load_checkpoint(path, policy, critic, optim=None) -> int # returns round n
def latest_checkpoint(cfg) -> tuple[int, Path] | None
def game_seed(cfg, round_n, actor_idx, game_idx) -> int
```

- [ ] **Step 1: Write the failing test**

`tests/test_trainloop.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.rollout import play_game
from ptcg.trainloop import (TrainConfig, checkpoint_path, latest_checkpoint,
                            load_checkpoint, load_round, round_dir, save_checkpoint,
                            save_game)


def test_game_roundtrip(tmp_path):
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    ep = play_game(m, (deck, list(deck)), tables, generator=g)
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    rd = round_dir(cfg, 0)
    rd.mkdir(parents=True)
    save_game(rd / "a0-g0.pt", ep)
    (eps,) = (load_round(cfg, 0),)
    assert len(eps) == 1
    e = eps[0]
    assert e.result == ep.result and e.rewards == ep.rewards
    assert len(e.steps) == len(ep.steps)
    assert e.steps[0].picks == ep.steps[0].picks
    assert float(e.steps[0].logprob) == float(ep.steps[0].logprob)
    assert (e.steps[0].state.numeric == ep.steps[0].state.numeric).all()


def test_checkpoint_roundtrip_and_latest(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny", seed=1)
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    save_checkpoint(cfg, 3, p, c, opt)
    n, path = latest_checkpoint(cfg)
    assert n == 3 and path == checkpoint_path(cfg, 3)
    p2 = PolicyModel(tiny_config(tables))
    c2 = CriticModel(critic_config(tables))
    opt2 = torch.optim.Adam(list(p2.parameters()) + list(c2.parameters()), lr=1e-3)
    assert load_checkpoint(path, p2, c2, opt2) == 3
    for a, b in zip(p.state_dict().values(), p2.state_dict().values()):
        assert torch.equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainloop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.trainloop'`

- [ ] **Step 3: Implement the IO half of `ptcg/trainloop.py`**

```python
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .featurize import FEATURIZER_VERSION
from .model import (CriticModel, PolicyModel, critic_config, student_config,
                    teacher_config, tiny_config)


@dataclass
class TrainConfig:
    run_dir: str
    model_size: str = "student"
    games_per_round: int = 192
    actors: int = 3
    epochs: int = 2
    minibatch: int = 512
    lr: float = 3e-4
    clip: float = 0.2
    lam: float = 0.95
    gamma: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    critic_coef: float = 0.5
    aux_coef: float = 0.1
    kl_stop: float = 0.02
    grad_clip: float = 1.0
    ratio_gate: float = 1e-3
    eval_every: int = 5
    eval_games_random: int = 200
    eval_games_ckpt: int = 100
    device: str = "cpu"
    seed: int = 0
    step_cap: int = 5000


def model_config_for(size, tables):
    return {"tiny": tiny_config, "student": student_config,
            "teacher": teacher_config}[size](tables)


def round_dir(cfg, n) -> Path:
    return Path(cfg.run_dir) / "rounds" / str(n)


def checkpoint_path(cfg, n) -> Path:
    return Path(cfg.run_dir) / f"checkpoint-{n:04d}.pt"


def game_seed(cfg, round_n, actor_idx, game_idx) -> int:
    return ((cfg.seed * 1_000_003 + round_n) * 101 + actor_idx) * 10_007 + game_idx


def save_game(path, episode) -> None:
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(episode, tmp)
    os.replace(tmp, path)


def load_round(cfg, n):
    rd = round_dir(cfg, n)
    eps = []
    for f in sorted(rd.glob("*.pt")):
        eps.append(torch.load(f, weights_only=False))
    return eps


def save_checkpoint(cfg, n, policy, critic, optim) -> None:
    path = checkpoint_path(cfg, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "round": n,
        "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
        "optim": optim.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "config": asdict(cfg),
        "featurizer_version": FEATURIZER_VERSION,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, critic, optim=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["policy"])
    critic.load_state_dict(ck["critic"])
    if optim is not None:
        optim.load_state_dict(ck["optim"])
        torch.set_rng_state(ck["torch_rng"])
    return ck["round"]


def latest_checkpoint(cfg):
    best = None
    for f in Path(cfg.run_dir).glob("checkpoint-*.pt"):
        n = int(f.stem.split("-")[1])
        if best is None or n > best[0]:
            best = (n, f)
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trainloop.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add ptcg/trainloop.py tests/test_trainloop.py
git commit -m "train config with trajectory and checkpoint io"
```

---

### Task 5: Actor processes and pool

**Files:**
- Create: `ptcg/actors.py`
- Test: `tests/test_actors.py`

**Interfaces:**
- Consumes: `play_game(priv_viz=True)`, `save_game/round_dir/game_seed/
  model_config_for/load_checkpoint/TrainConfig` (Task 4), `build_tables`,
  `load_sample_deck`, `PolicyModel`, `random_picks`, `BattleSession`.
- Produces:

```python
def collect_round_worker(args: tuple) -> dict
    # args = (cfg_json: str, round_n: int, actor_idx: int, n_games: int,
    #         ckpt_path: str). Top-level function (spawn-safe). Opens its own
    #         engine; plays n_games mirror games with priv_viz=True; saves each
    #         to round_dir as f"a{actor_idx}-g{g}.pt"; returns
    #         {"games": int, "steps": int, "results": [int, ...], "wall_s": float}
def run_actor_pool(cfg, round_n, ckpt_path, worker=collect_round_worker,
                   extra=None) -> list[dict]
    # splits cfg.games_per_round across cfg.actors processes (spawn),
    # remainder to the first actors; waits for all; returns their stats
def play_versus(model, opponent, tables, decks, generator, model_seat) -> int
    # one game: `model` samples on model_seat; `opponent` is "random" or a
    # PolicyModel for the other seat. Returns 1 if model_seat won, 0 if lost
    # or drew. One BattleSession per call.
```

- [ ] **Step 1: Write the failing test**

`tests/test_actors.py`:

```python
import json
import torch
from ptcg.actors import collect_round_worker, play_versus, run_actor_pool
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, load_round, round_dir, save_checkpoint
from dataclasses import asdict


def _seed_run(tmp_path, games):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                      games_per_round=games, actors=1)
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()), lr=1e-3)
    save_checkpoint(cfg, 0, p, c, opt)
    return tables, cfg, p


def test_collect_round_worker_writes_games(tmp_path):
    tables, cfg, _ = _seed_run(tmp_path, 2)
    stats = collect_round_worker(
        (json.dumps(asdict(cfg)), 0, 0, 2, str(tmp_path / "checkpoint-0000.pt")))
    assert stats["games"] == 2 and stats["steps"] >= 10
    eps = load_round(cfg, 0)
    assert len(eps) == 2
    assert all(5 <= len(e.steps) <= 1000 for e in eps)
    # priv_viz was on: some step in some game carries opp deck entity tokens
    from ptcg.featurize import KIND_ENTITY, OWNER_OPP
    found = any(
        any((s.priv_state.zone[i] == 1 and s.priv_state.owner[i] == OWNER_OPP
             and s.priv_state.kind[i] == KIND_ENTITY)
            for i in range(s.priv_state.n))
        for e in eps for s in e.steps)
    assert found


def test_play_versus_random(tmp_path):
    tables, cfg, p = _seed_run(tmp_path, 1)
    deck = load_sample_deck()
    g = torch.Generator().manual_seed(9)
    r = play_versus(p, "random", tables, (deck, list(deck)), g, model_seat=0)
    assert r in (0, 1)
```

(`run_actor_pool` with real subprocesses is exercised in Task 7's integration
test; unit-testing spawn here would double engine startup cost for no extra
coverage.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_actors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ptcg.actors'`

- [ ] **Step 3: Implement `ptcg/actors.py`**

```python
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .cards import build_tables
from .engine import BattleSession, load_sample_deck, random_picks
from .featurize import encode_select, featurize_state
from .model import PolicyModel
from .rollout import play_game
from .tracker import BeliefTracker
from .trainloop import (TrainConfig, game_seed, load_checkpoint,
                        model_config_for, round_dir)
from .action import sample_select


class _NullCritic:
    """Checkpoint files carry the critic too; actors don't need it."""

    def load_state_dict(self, sd):
        return None


def collect_round_worker(args):
    cfg_json, round_n, actor_idx, n_games, ckpt_path = args
    cfg = TrainConfig(**json.loads(cfg_json))
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(model_config_for(cfg.model_size, tables))
    load_checkpoint(ckpt_path, policy, _NullCritic(), optim=None)
    policy.eval()
    rd = round_dir(cfg, round_n)
    rd.mkdir(parents=True, exist_ok=True)
    from .trainloop import save_game
    t0 = time.perf_counter()
    steps = 0
    results = []
    for g in range(n_games):
        gen = torch.Generator().manual_seed(
            game_seed(cfg, round_n, actor_idx, g))
        # spec's persistent debug sample: raw obs of the first game of every
        # 10th round, captured by actor 0 only
        obs_log = [] if (round_n % 10 == 0 and actor_idx == 0 and g == 0) else None
        with torch.no_grad():
            ep = play_game(policy, (deck, list(deck)), tables, generator=gen,
                           step_cap=cfg.step_cap, priv_viz=True,
                           obs_log=obs_log)
        save_game(rd / f"a{actor_idx}-g{g}.pt", ep)
        if obs_log is not None:
            dbg = Path(cfg.run_dir) / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            torch.save(obs_log, dbg / f"round-{round_n:04d}-g0-obs.pt")
        steps += len(ep.steps)
        results.append(ep.result)
    return {"games": n_games, "steps": steps, "results": results,
            "wall_s": time.perf_counter() - t0}


def run_actor_pool(cfg, round_n, ckpt_path, worker=collect_round_worker,
                   extra=None):
    import multiprocessing as mp
    per = cfg.games_per_round // cfg.actors
    rem = cfg.games_per_round - per * cfg.actors
    jobs = []
    for a in range(cfg.actors):
        n = per + (1 if a < rem else 0)
        if n == 0:
            continue
        base = (json.dumps(asdict(cfg)), round_n, a, n, str(ckpt_path))
        jobs.append(base + tuple(extra or ()))
    if len(jobs) == 1:
        return [worker(jobs[0])]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        return pool.map(worker, jobs)


def play_versus(model, opponent, tables, decks, generator, model_seat):
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    rng = random.Random(int(torch.randint(1 << 30, (1,), generator=generator)))
    try:
        n = 0
        while not s.done:
            n += 1
            if n > 5000:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            trackers[me].update(s.obs.get("logs", []))
            actor = model if me == model_seat else opponent
            if actor == "random":
                s.select(random_picks(s.obs, rng))
                continue
            ts = featurize_state(s.obs, me, decks[me],
                                 trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            with torch.no_grad():
                d = sample_select(actor, ts, es, generator)
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    return 1 if r == model_seat else 0
```


- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_actors.py -v`
Expected: 2 PASS (a couple of minutes: two full tiny-model games + one eval game)

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m pytest tests/ -q` → all pass.

```bash
git add ptcg/actors.py tests/test_actors.py
git commit -m "actor processes with pool and versus play"
```

---

### Task 6: Learner update

**Files:**
- Modify: `ptcg/trainloop.py` (append)
- Test: `tests/test_learner.py`

**Interfaces:**
- Consumes: `batched_replay/replay_logprob_batched` (Task 1), `ppo.py`
  (Task 2), IO from Task 4.
- Produces:

```python
def learner_update(policy, critic, optim, episodes, cfg, tables,
                   opp_deck) -> dict
    # runs the ratio gate + up to cfg.epochs PPO epochs; returns metrics:
    # {"loss_pg","loss_v","loss_critic","loss_aux","entropy","approx_kl",
    #  "epochs_ran","ratio_drift","steps"}
    # raises RuntimeError if epoch-0 max|ratio-1| > cfg.ratio_gate
```

- [ ] **Step 1: Write the failing test**

`tests/test_learner.py`:

```python
import torch
from ptcg.cards import build_tables
from ptcg.engine import load_sample_deck
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.rollout import play_game
from ptcg.trainloop import TrainConfig, learner_update


def test_learner_update_ratio_gate_and_metrics():
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=3e-4)
    g = torch.Generator().manual_seed(6)
    eps = [play_game(policy, (deck, list(deck)), tables, generator=g,
                     priv_viz=True) for _ in range(2)]
    cfg = TrainConfig(run_dir="unused", model_size="tiny", epochs=2,
                      minibatch=64, device="cpu")
    before = [p.clone() for p in policy.parameters()]
    m = learner_update(policy, critic, optim, eps, cfg, tables, deck)
    # same weights collected the data: drift must be ~0 (same device/config)
    assert m["ratio_drift"] < 1e-6
    assert m["epochs_ran"] >= 1 and m["steps"] > 10
    for k in ("loss_pg", "loss_v", "loss_critic", "loss_aux",
              "entropy", "approx_kl"):
        assert torch.isfinite(torch.tensor(m[k])), k
    assert any(not torch.equal(a, b)
               for a, b in zip(before, policy.parameters()))


def test_learner_update_aborts_on_stale_policy():
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(tiny_config(tables))
    critic = CriticModel(critic_config(tables))
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(7)
    eps = [play_game(policy, (deck, list(deck)), tables, generator=g,
                     priv_viz=True)]
    other = PolicyModel(tiny_config(tables))  # different weights entirely
    cfg = TrainConfig(run_dir="unused", model_size="tiny", device="cpu")
    import pytest
    with pytest.raises(RuntimeError):
        learner_update(other, critic, optim, eps, cfg, tables, deck)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learner.py -v`
Expected: FAIL with `ImportError: cannot import name 'learner_update'`

- [ ] **Step 3: Implement `learner_update`** (append to `ptcg/trainloop.py`)

```python
def learner_update(policy, critic, optim, episodes, cfg, tables, opp_deck):
    import torch as _t
    from .model import collate_selects, collate_states
    from .ppo import (assemble_advantages, aux_targets, ppo_policy_loss)
    from .replay import batched_replay

    device = _t.device(cfg.device)
    policy.to(device)
    critic.to(device)
    steps, old_lp, adv, ret = assemble_advantages(
        episodes, critic, device=device, lam=cfg.lam, gamma=cfg.gamma)
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_deck)
    old_lp, adv, ret = old_lp.to(device), adv.to(device), ret.to(device)
    pd_t, dl_t, hd_t = pd_t.to(device), dl_t.to(device), hd_t.to(device)
    B = len(steps)

    def minibatches(order):
        for lo in range(0, B, cfg.minibatch):
            yield order[lo:lo + cfg.minibatch]

    def replay_mb(idx, grad):
        sb = collate_states([steps[i].state for i in idx])
        selb = collate_selects([steps[i].esel for i in idx])
        sb = {k: v.to(device) for k, v in sb.items()}
        selb = {k: v.to(device) for k, v in selb.items()}
        ctx = _t.enable_grad() if grad else _t.no_grad()
        with ctx:
            trunk = policy.encode(sb)
            lp, ent = batched_replay(policy, trunk, sb, selb,
                                     [steps[i].picks for i in idx])
        return sb, trunk, lp, ent

    # epoch-0 ratio gate over the full round
    with _t.no_grad():
        drifts = []
        for idx in minibatches(list(range(B))):
            _, _, lp, _ = replay_mb(idx, grad=False)
            drifts.append((lp - old_lp[idx]).exp().sub(1).abs().max())
        ratio_drift = float(_t.stack(drifts).max()) if drifts else 0.0
    if ratio_drift > cfg.ratio_gate:
        raise RuntimeError(
            f"ratio gate violated: max|ratio-1|={ratio_drift:.2e} "
            f"> {cfg.ratio_gate:.0e} — policy/data mismatch")

    poiss = _t.nn.PoissonNLLLoss(log_input=False, full=False)
    agg = {k: 0.0 for k in ("loss_pg", "loss_v", "loss_critic", "loss_aux",
                            "entropy", "approx_kl")}
    n_mb = 0
    epochs_ran = 0
    gen = _t.Generator().manual_seed(cfg.seed + 17)
    for epoch in range(cfg.epochs):
        order = _t.randperm(B, generator=gen).tolist()
        kl_epoch = []
        for idx in minibatches(order):
            sb, trunk, lp, ent = replay_mb(idx, grad=True)
            pg, ratio, kl = ppo_policy_loss(
                lp, old_lp[idx], adv[idx], clip=cfg.clip)
            v = policy.public_value(trunk)
            vloss = ((v - ret[idx]) ** 2).mean()
            pvb = collate_states([steps[i].priv_state for i in idx])
            pvb = {k: t.to(device) for k, t in pvb.items()}
            cvals = critic(pvb)
            players = _t.tensor([steps[i].player for i in idx], device=device)
            closs = ((cvals[_t.arange(len(idx), device=device), players]
                      - ret[idx]) ** 2).mean()
            aux = (((policy.prize_diff(trunk) - pd_t[idx]) ** 2).mean()
                   + poiss(policy.aux_decklist(trunk), dl_t[idx])
                   + poiss(policy.aux_hand(trunk), hd_t[idx]))
            loss = (pg + cfg.vf_coef * vloss + cfg.critic_coef * closs
                    - cfg.ent_coef * ent.mean() + cfg.aux_coef * aux)
            optim.zero_grad()
            loss.backward()
            _t.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()),
                cfg.grad_clip)
            optim.step()
            for k, val in (("loss_pg", pg), ("loss_v", vloss),
                           ("loss_critic", closs), ("loss_aux", aux),
                           ("entropy", ent.mean()), ("approx_kl", kl)):
                agg[k] += float(val.detach())
            kl_epoch.append(float(kl.detach()))
            n_mb += 1
        epochs_ran += 1
        if kl_epoch and sum(kl_epoch) / len(kl_epoch) > cfg.kl_stop:
            break
    out = {k: v / max(n_mb, 1) for k, v in agg.items()}
    out.update(epochs_ran=epochs_ran, ratio_drift=ratio_drift, steps=B)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learner.py -v`
Expected: 2 PASS (a few minutes: real tiny games + updates on CPU)

- [ ] **Step 5: Commit**

```bash
git add ptcg/trainloop.py tests/test_learner.py
git commit -m "learner update with ratio gate and kl stop"
```

---

### Task 7: Orchestrator, metrics, resume

**Files:**
- Modify: `ptcg/trainloop.py` (append)
- Test: `tests/test_train_integration.py`

**Interfaces:**
- Consumes: everything above; `run_actor_pool` (Task 5).
- Produces:

```python
METRIC_FIELDS = ["round", "kind", "games", "steps", "loss_pg", "loss_v",
                 "loss_critic", "loss_aux", "entropy", "approx_kl",
                 "epochs_ran", "ratio_drift", "wr_random", "ci_random",
                 "wr_ck5", "wr_ck15", "mean_len", "wall_s"]
def append_metrics(cfg, row: dict) -> None      # csv append, header once
def read_metrics(cfg) -> list[dict]
def truncate_metrics(cfg, before_round: int) -> None  # drop rows >= round
def train(cfg, max_rounds: int) -> None
    # fresh run: writes config.json + checkpoint-0000; resume: loads latest
    # checkpoint, truncates metrics at that round, continues. Per round:
    # run_actor_pool -> load_round -> learner_update -> save_checkpoint ->
    # append_metrics -> shutil.rmtree(round dir) -> eval when due (Task 8
    # wires the eval; until then the eval hook is a no-op `_eval_due` stub
    # returning None).
```

- [ ] **Step 1: Write the failing test**

`tests/test_train_integration.py`:

```python
from ptcg.trainloop import (TrainConfig, latest_checkpoint, read_metrics,
                            round_dir, train)


def _cfg(tmp_path):
    return TrainConfig(run_dir=str(tmp_path), model_size="tiny",
                       games_per_round=2, actors=1, epochs=1, minibatch=64,
                       eval_every=999, device="cpu", seed=5)


def test_one_round_end_to_end_then_resume(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg, max_rounds=1)
    n, _ = latest_checkpoint(cfg)
    assert n == 1
    rows = read_metrics(cfg)
    assert len(rows) == 1 and rows[0]["kind"] == "train"
    assert int(rows[0]["round"]) == 0 and int(rows[0]["steps"]) > 5
    assert not round_dir(cfg, 0).exists()  # consumed and deleted

    train(cfg, max_rounds=2)  # resume: must NOT redo round 0
    n2, _ = latest_checkpoint(cfg)
    assert n2 == 2
    rows = read_metrics(cfg)
    assert [int(r["round"]) for r in rows if r["kind"] == "train"] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_integration.py -v`
Expected: FAIL with `ImportError: cannot import name 'train'`

- [ ] **Step 3: Implement** (append to `ptcg/trainloop.py`)

```python
import csv
import shutil

METRIC_FIELDS = ["round", "kind", "games", "steps", "loss_pg", "loss_v",
                 "loss_critic", "loss_aux", "entropy", "approx_kl",
                 "epochs_ran", "ratio_drift", "wr_random", "ci_random",
                 "wr_ck5", "wr_ck15", "mean_len", "wall_s"]


def _metrics_path(cfg):
    return Path(cfg.run_dir) / "metrics.csv"


def append_metrics(cfg, row):
    p = _metrics_path(cfg)
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def read_metrics(cfg):
    p = _metrics_path(cfg)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def truncate_metrics(cfg, before_round):
    rows = [r for r in read_metrics(cfg) if int(r["round"]) < before_round]
    p = _metrics_path(cfg)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _eval_due(cfg, round_n, policy, tables):
    return None  # wired in Task 8


def train(cfg, max_rounds):
    import time as _time

    from .actors import run_actor_pool
    from .cards import build_tables
    from .engine import load_sample_deck

    tables = build_tables()
    deck = load_sample_deck()
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    policy = PolicyModel(model_config_for(cfg.model_size, tables))
    critic = CriticModel(critic_config(tables))
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=cfg.lr)

    latest = latest_checkpoint(cfg)
    if latest is None:
        torch.manual_seed(cfg.seed)
        with open(Path(cfg.run_dir) / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=1)
        save_checkpoint(cfg, 0, policy, critic, optim)
        start = 0
    else:
        start = load_checkpoint(latest[1], policy, critic, optim)
        truncate_metrics(cfg, start)
        rd = round_dir(cfg, start)
        if rd.exists():
            shutil.rmtree(rd)  # incomplete round from a crash

    for rnd in range(start, max_rounds):
        t0 = _time.perf_counter()
        ck = checkpoint_path(cfg, rnd)
        stats = run_actor_pool(cfg, rnd, ck)
        episodes = load_round(cfg, rnd)
        policy.train()
        critic.train()
        m = learner_update(policy, critic, optim, episodes, cfg, tables, deck)
        policy.eval()
        critic.eval()
        save_checkpoint(cfg, rnd + 1, policy, critic, optim)
        n_steps = sum(s["steps"] for s in stats)
        mean_len = n_steps / max(sum(s["games"] for s in stats), 1)
        append_metrics(cfg, dict(
            round=rnd, kind="train", games=sum(s["games"] for s in stats),
            mean_len=f"{mean_len:.1f}", wall_s=f"{_time.perf_counter() - t0:.0f}",
            **{k: (f"{v:.6g}" if isinstance(v, float) else v)
               for k, v in m.items()}))
        shutil.rmtree(round_dir(cfg, rnd), ignore_errors=True)
        ev = _eval_due(cfg, rnd, policy, tables)
        if ev is not None:
            append_metrics(cfg, ev)
```

Note: `train` deliberately takes `max_rounds` (tests and bounded runs);
`scripts/train.py` (Task 9) defaults it to a very large number.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_integration.py -v`
Expected: 1 PASS (~2–5 minutes: four tiny games plus two updates, single
process — `actors=1` avoids spawn inside pytest; spawn is covered by the
Task 9 smoke run)

- [ ] **Step 5: Run the full suite, commit**

Run: `python -m pytest tests/ -q` → all pass.

```bash
git add ptcg/trainloop.py tests/test_train_integration.py
git commit -m "round orchestrator with metrics and resume"
```

---

### Task 8: Evaluation pass

**Files:**
- Modify: `ptcg/actors.py` (append `eval_worker`), `ptcg/trainloop.py`
  (replace `_eval_due` stub)
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces:

```python
# actors.py
def eval_worker(args: tuple) -> dict
    # args = (cfg_json, round_n, actor_idx, n_games, ckpt_path,
    #         opponent_ckpt_path_or_"random")
    # plays n_games of current policy vs opponent, alternating model_seat by
    # game parity; returns {"wins": int, "games": int}
# trainloop.py — replaces the stub
def _eval_due(cfg, round_n, policy, tables) -> dict | None
    # None unless (round_n + 1) % cfg.eval_every == 0. Otherwise runs, via
    # run_actor_pool(worker=eval_worker, ...): cfg.eval_games_random games vs
    # "random", and cfg.eval_games_ckpt games vs checkpoints round_n-4 and
    # round_n-14 when those files exist. Returns a metrics row dict with
    # kind="eval", wr_random, ci_random (1.96*sqrt(p(1-p)/n)), wr_ck5, wr_ck15.
```

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:

```python
import json
from dataclasses import asdict

import torch
from ptcg.actors import eval_worker
from ptcg.cards import build_tables
from ptcg.model import CriticModel, PolicyModel, critic_config, tiny_config
from ptcg.trainloop import TrainConfig, save_checkpoint


def test_eval_worker_vs_random(tmp_path):
    tables = build_tables()
    cfg = TrainConfig(run_dir=str(tmp_path), model_size="tiny")
    p = PolicyModel(tiny_config(tables))
    c = CriticModel(critic_config(tables))
    opt = torch.optim.Adam(list(p.parameters()) + list(c.parameters()))
    save_checkpoint(cfg, 0, p, c, opt)
    out = eval_worker((json.dumps(asdict(cfg)), 0, 0, 2,
                       str(tmp_path / "checkpoint-0000.pt"), "random"))
    assert out["games"] == 2 and 0 <= out["wins"] <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'eval_worker'`

- [ ] **Step 3: Implement**

Append to `ptcg/actors.py`:

```python
def eval_worker(args):
    cfg_json, round_n, actor_idx, n_games, ckpt_path, opp_spec = args
    cfg = TrainConfig(**json.loads(cfg_json))
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(model_config_for(cfg.model_size, tables))
    load_checkpoint(ckpt_path, policy, _NullCritic())
    policy.eval()
    if opp_spec == "random":
        opponent = "random"
    else:
        opponent = PolicyModel(model_config_for(cfg.model_size, tables))
        load_checkpoint(opp_spec, opponent, _NullCritic())
        opponent.eval()
    wins = 0
    for g in range(n_games):
        gen = torch.Generator().manual_seed(
            game_seed(cfg, 100_000 + round_n, actor_idx, g))
        wins += play_versus(policy, opponent, tables, (deck, list(deck)),
                            gen, model_seat=g % 2)
    return {"wins": wins, "games": n_games}
```

Replace the `_eval_due` stub in `ptcg/trainloop.py`:

```python
def _eval_due(cfg, round_n, policy, tables):
    if (round_n + 1) % cfg.eval_every != 0:
        return None
    import math

    from .actors import eval_worker, run_actor_pool
    ck = checkpoint_path(cfg, round_n + 1)
    row = {"round": round_n, "kind": "eval"}

    def _run(n_games, opp_spec):
        sub = TrainConfig(**{**asdict(cfg), "games_per_round": n_games})
        stats = run_actor_pool(sub, round_n, ck, worker=eval_worker,
                               extra=(opp_spec,))
        wins = sum(s["wins"] for s in stats)
        games = sum(s["games"] for s in stats)
        return wins / max(games, 1), games

    wr, n = _run(cfg.eval_games_random, "random")
    row["wr_random"] = f"{wr:.3f}"
    row["ci_random"] = f"{1.96 * math.sqrt(max(wr * (1 - wr), 1e-9) / max(n, 1)):.3f}"
    for label, back in (("wr_ck5", 5), ("wr_ck15", 15)):
        ref = checkpoint_path(cfg, round_n + 1 - back)
        if ref.exists():
            wr, _ = _run(cfg.eval_games_ckpt, str(ref))
            row[label] = f"{wr:.3f}"
    return row
```

Note `run_actor_pool` already forwards `extra` into each job tuple (Task 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval.py -v` → 1 PASS
Run: `python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add ptcg/actors.py ptcg/trainloop.py tests/test_eval.py
git commit -m "eval pass vs random and past checkpoints"
```

---

### Task 9: CLI, plots, smoke runs

**Files:**
- Create: `scripts/train.py`, `scripts/plot_run.py`,
  `benchmarks/RESULTS-train-smoke.md`
- Modify: `pyproject.toml` (add `matplotlib` to dev extras)

**Interfaces:**
- Consumes: `train/TrainConfig/read_metrics` (Task 7).
- Produces: the operator entry points for the phase-2 qualification run.

- [ ] **Step 1: Write `scripts/train.py`**

```python
"""Phase-2 training entry. Run from the repo root with the training venv:
venv-train\\Scripts\\python scripts\\train.py --run-id phase2-a --device cuda
"""
import argparse
from dataclasses import fields

from ptcg.trainloop import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-rounds", type=int, default=1_000_000)
    for f in fields(TrainConfig):
        if f.name == "run_dir":
            continue
        t = type(f.default)
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=t,
                        default=f.default)
    args = ap.parse_args()
    kw = {f.name: getattr(args, f.name) for f in fields(TrainConfig)
          if f.name != "run_dir"}
    cfg = TrainConfig(run_dir=f"runs/{args.run_id}", **kw)
    train(cfg, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/plot_run.py`**

```python
"""Plot metrics.csv curves: python scripts/plot_run.py runs/<run-id>"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir):
    import csv
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv", newline="")))
    train_rows = [r for r in rows if r["kind"] == "train"]
    eval_rows = [r for r in rows if r["kind"] == "eval"]
    out = Path(run_dir) / "plots"
    out.mkdir(exist_ok=True)

    def series(rs, key):
        pts = [(int(r["round"]), float(r[key])) for r in rs if r.get(key)]
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, keys, title in (
            (axes[0][0], ["loss_pg", "loss_v", "loss_critic"], "losses"),
            (axes[0][1], ["entropy"], "entropy"),
            (axes[1][0], ["approx_kl", "ratio_drift"], "kl / drift"),
            (axes[1][1], ["mean_len"], "game length")):
        for k in keys:
            ax.plot(*series(train_rows, k), label=k)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "train.png", dpi=120)

    if eval_rows:
        fig2, ax = plt.subplots(figsize=(8, 5))
        for k in ("wr_random", "wr_ck5", "wr_ck15"):
            x, y = series(eval_rows, k)
            if x:
                ax.plot(x, y, marker="o", label=k)
        ax.axhline(0.5, ls="--", c="gray")
        ax.axhline(0.65, ls=":", c="green")
        ax.set_ylim(0, 1)
        ax.set_title("win rates")
        ax.legend()
        fig2.savefig(out / "eval.png", dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
```

Add `"matplotlib>=3.8"` to `[project.optional-dependencies] dev` in
`pyproject.toml`.

- [ ] **Step 3: CPU smoke run (base interpreter, spawn path included)**

Run: `python scripts/train.py --run-id smoke-cpu --model-size tiny --games-per-round 6 --actors 2 --max-rounds 2 --minibatch 64 --epochs 1 --eval-every 2 --eval-games-random 4 --eval-games-ckpt 2 --device cpu`
Expected: completes in ~5–15 minutes; `runs/smoke-cpu/` contains
`checkpoint-0000..0002.pt`, `metrics.csv` with 2 train rows + 1 eval row,
no `rounds/` residue. This exercises real spawn multiprocessing (2 actors).
Then: `python -m pip install matplotlib` (base env, if missing) and
`python scripts/plot_run.py runs/smoke-cpu` → `plots/train.png`, `plots/eval.png`.

- [ ] **Step 4: GPU smoke run (training venv)**

Run: `venv-train\Scripts\python scripts\train.py --run-id smoke-gpu --model-size student --games-per-round 12 --actors 3 --max-rounds 1 --minibatch 256 --eval-every 999 --device cuda`
(Use `--device cpu` here if Task 0 recorded the CPU fallback.)
Expected: completes; metrics row shows finite losses and `ratio_drift`
below 1e-3 (this is the first real cross-config CPU-actor/GPU-learner
exactness measurement — record the value).

- [ ] **Step 5: Record and commit**

Write `benchmarks/RESULTS-train-smoke.md`: date, both smoke commands, wall
times, games/steps, the measured GPU `ratio_drift`, and actor throughput
(games/hour extrapolated). Delete `runs/smoke-*` afterwards (they are
git-ignored anyway).

```bash
git add scripts/train.py scripts/plot_run.py pyproject.toml benchmarks/RESULTS-train-smoke.md
git commit -m "training cli with plots and smoke runs"
```

---

### Task 10: Qualification run playbook (operator step)

**Files:**
- Create: `docs/phase2-run.md`

This task launches the multi-day run and defines how it is judged. It is an
operator playbook, not new code.

- [ ] **Step 1: Write `docs/phase2-run.md`** containing exactly:

```markdown
# Phase-2 qualification run

Launch (training venv, repo root):

    venv-train\Scripts\python scripts\train.py --run-id phase2-a --device cuda --seed 1

(`--device cpu` if benchmarks/RESULTS-gpu.md recorded the CPU fallback.)
Resume after any interruption: run the same command — the loop continues
from the last complete checkpoint.

Monitor (any interpreter):

    python scripts/plot_run.py runs/phase2-a

Success criteria (spec §Success criteria — all three required, ≤72 h run):
1. wr_random ≥ 0.65 on two consecutive eval rows.
2. wr_ck15 > 0.55 on three consecutive eval rows.
3. Zero ratio-gate aborts, zero NaN aborts, and at least one kill+resume
   with contiguous metrics.

On success: record the run id, final eval row, and wall-clock in
benchmarks/RESULTS-train-smoke.md under a "qualification" heading, and
phase 2 is complete. On failure: capture metrics.csv and plots, and open a
debugging session — do not reinterpret the criteria.
```

- [ ] **Step 2: Launch the run** (only if the session has the machine to
  itself — otherwise hand the command to the human): start the command from
  the playbook in the background, verify `runs/phase2-a/metrics.csv` gains a
  first train row (~1 h at student size), then perform one kill+resume cycle
  early (criterion 3) and confirm the round counter continues without a gap.

- [ ] **Step 3: Commit**

```bash
git add docs/phase2-run.md
git commit -m "phase 2 qualification run playbook"
```

---

## Self-review notes (kept for the record)

- Spec coverage: two-venv + Pascal spike → Task 0; batched replay + ratio
  refinement → Task 1 (gate enforced in Task 6); PPO defaults/GAE/aux → Tasks
  2, 6; viz deck order to the critic → Task 3; round lifecycle, storage,
  deletion, resume → Tasks 4, 7; actors/mirror self-play/seeding → Task 5;
  eval cadence and reference checkpoints → Task 8; metrics/plots/CLI → Tasks
  7, 9; success criteria + long run → Task 10; debug sample of raw obs →
  `obs_log` hook (Task 3) + actor-0 capture of the first game of every 10th
  round into `<run_dir>/debug/` (Task 5).
- Type consistency: `TrainConfig` fields match `scripts/train.py` flag
  generation; `run_actor_pool(worker=, extra=)` matches both workers'
  arg-tuple shapes; `batched_replay(model, trunk, sb, selb, picks_list)`
  matches Task 6's call; metrics field names match `plot_run.py` keys.
- Known risks called out in-task: batched-vs-B1 float exactness (Task 1
  decision rule), spawn under pytest avoided (Task 7 note), Pascal wheel
  ladder with CPU fallback (Task 0).
```
