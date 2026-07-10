# Expert iteration from search — phase 5 (design)

- Date: 2026-07-10
- Status: approved in brainstorming
- Scope: break the phase-3 self-play plateau by closing the loop phase 4 opened —
  generate self-play games where BOTH seats use the phase-4 search, train the policy
  toward the search's root visit distributions and the value head toward outcomes,
  gate each cycle, and ship promoted nets as drop-in `policy.pt` replacements in the
  UNCHANGED phase-4 Kaggle agent. Cloud-rented CPU-heavy boxes + one mid-tier GPU;
  staged spend inside a ~$2k budget over ~1 month.
- Builds on: phase-3 (`2026-07-05-league-deck-portfolio-phase3-design.md`, converged
  generalist), phase-4 (`2026-07-09-inference-search-phase4-design.md`, the search and
  its amended phase-consistent leaf rule).

## Why this, why now (evidence)

- The phase-3 plateau was a plateau OF the 1060-shaped system: d224 capacity (chosen
  for 6 GiB VRAM), ~1,350 games/hr data rate, one homogenized league (wr_ck5/15
  pinned at 0.50). It is not evidence that the problem is exhausted.
- Phase 4 PROVED headroom above the plateau: search beats the converged raw policy
  0.565 [0.516,0.613] n=400 at 0.8 s/move and 0.680 [0.583,0.763] n=100 at 3 s/move.
  A measured policy-improvement operator exists; expert iteration (the AlphaGo-Zero /
  AlphaZero loop) distills it back into the net, which strengthens the operator itself
  on the next cycle.
- Phase 4 also identified a turn-phase calibration artifact in `public_value`
  (turn-start |v|≈0.9 vs ~0.6 mid-turn for comparable advantage). Training the value
  head on search-play outcomes across ALL states (turn-starts included) re-anchors it —
  and better leaf values directly strengthen the search.
- Hardware honesty: 8×B200 is NOT the win-rate-per-dollar optimum here. Generation is
  CPU-simulator-bound (~1–1.5 s per searched move per core at d224 in-process); the
  nets are small; a 64–128 vCPU box + one 4090/A100-class GPU covers stage 1. B200-class
  hardware only earns its premium under stage-2 capacity scaling, if triggered.

## User-set constraints

- Compute: hourly rentals (provider flexible), budget ~$500–2k, ceiling $2k.
- Timeline: ~1 month until submissions effectively lock; mid-month and end-of-month
  Kaggle re-submission checkpoints.
- Deployment target unchanged: Kaggle CPU, 600 s overage bank, the validated phase-4
  agent — a promoted net ships as a new `policy.pt` only; agent/search/packaging code
  do not change in this phase.

## Architecture: the cycle

One cycle = generate → train → gate → promote, orchestrated by a resumable driver.

### 1. Generate — `ptcg/selfplay_search.py` (new)

- Search-vs-search self-play over the existing 9-deck portfolio mix (mirrors +
  cross-pairs sampled like the phase-3 league worker; per-game seeds/decks recorded).
- Both seats use the UNCHANGED phase-4 `search_move`, driven by a **sims budget, not
  wall-clock**: pass `tslice=float("inf")` so the existing `k_trees × sims_per_tree`
  ceiling binds (e.g. K=3 × 32 = 96 sims/move) — no search-code change needed; data
  strength is machine-load-independent and reproducible-ish.
- Recorded per move (reusing the phase-2 Episode/step tensor format so existing
  collate/replay code applies): featurized state (ts/es), privileged state (for aux
  targets), acting seat, the root's candidate pick-tuples, and the normalized
  cross-tree visit distribution π over those candidates. Per game: outcome z (±1/0),
  decks, seeds.
- Generation runs many independent worker processes (one battle per process at a time,
  sequential games per worker — the established pattern).

### 2. Train — `ptcg/ei.py` + `scripts/train_ei.py` (new; supervised, no PPO)

- Policy loss: cross-entropy to π. Single-pick selects (98% of moves): one
  `option_logits` pass, softmax over options + done column (done carries π(decline)
  when minCount==0). Multi-pick selects: −Σ_a π(a)·logprob(sequence a) over the ≤~9
  recorded candidates via the existing forced-replay path.
- Value loss: MSE(public_value, z) from the acting seat, on EVERY recorded state
  including turn-starts (the calibration fix).
- Aux losses: decklist/hand Poisson targets from privileged data, unchanged — the
  determinizer's accuracy is a search input; letting these decay would weaken every
  later cycle.
- Stability: fresh cycle data mixed ~3:1 with a replay buffer of the previous 1–2
  cycles; optional small KL-to-incumbent penalty for the first cycle or two, dropped
  once cycles stabilize; π temperature knob against overly-sharp targets (default 1.0,
  raised only on evidence from entropy monitoring); AdamW, modest constant LR, 1–4
  epochs per cycle.
- The privileged CriticModel and PPO/GAE machinery are unused in this loop (kept in
  the repo, untouched).

### 3. Gate — `scripts/gate_ei.py` (new)

Candidate vs incumbent, in order:
1. Raw-policy head-to-head (fast smoke; directional).
2. **Search-wrapped head-to-head at fixed sims, n≥300, promotion requires the Wilson
   95% CI to clear 0.50** — the metric that matters, since search-wrapped is what ships.
3. Anchor suite (no-regression): vs frozen phase-3 generalist (raw + search-wrapped),
   vs random, and per-deck cross-deck checks — catches self-play meta drift.
4. Observability (non-gating): the phase-4 turn-boundary value-jump audit statistic,
   tracked per cycle (expected to shrink); policy entropy; value calibration curves.

### 4. Promote & loop — `scripts/ei_loop.py` (new)

- Cycle state (checkpoint, data manifest, gate verdict, metrics) written per cycle;
  re-running the driver is idempotent per cycle; artifacts rsync to local + storage so
  a vanished rental box costs at most the in-flight step.
- A failed gate is data, not disaster: diagnose (entropy, calibration audit, buffer
  ratio), adjust ONE variable, re-run the cycle (~tens of dollars each).

## Infrastructure

- Stage-1 box: 64–128 vCPU + 1× 4090/A100-class GPU, Linux x86 (the competition's
  native `libcg.so` runs there; `PTCG_ENGINE_DIR` mechanism as today). Provider
  flexible; `scripts/cloud_setup.sh` pins the environment.
- Throughput model (measured basis): ~1–1.5 s per searched move per core at d224
  in-process CPU inference → 64 cores ≈ 3–5k games/hr → a 20–40k-game cycle (~1M
  moves, ~2–6 GB tensors) ≈ $15–25 generation + a few dollars training/gating.
- Local 1060: stage-0 validation, gate spot-checks, packaging/ship (unchanged
  phase-4 pipeline).
- Long-running cloud processes: launched detached on the box (never as children of
  agent sessions — phase-2 lesson).

## Stages (pre-registered triggers, ~$2k ceiling)

- **Stage 0 (local + ≤$50):** build recorder/trainer/gate/driver; one miniature
  end-to-end cycle (~1k games, small sims). Success bar = mechanism works (targets
  flow, losses drop, gates run) — NOT strength.
- **Stage 1 (~$100–500):** 3–6 full cycles at d224, promotion per the gate.
- **Stage 2 (only if triggered; ~$300–800):** trigger = cycles still pass gates but
  per-cycle search-wrapped gains fall below ~+2% — capacity likely binding. Then:
  d384/d512 net initialized by distilling from the stage-1 net, continue cycles,
  **distill back to a CPU-fast student** before shipping (per-move latency envelope
  re-measured with the phase-4 clock before any upload). A GPU inference server for
  generation is a stage-2 contingency only.
- **Ship checkpoints (mid-month + end):** promoted net → `extract_policy` →
  `policy.pt` → unchanged phase-4 packaging (`make_submission.py`, both zips) →
  local search-vs-search gate against the CURRENT submission's net → user uploads.

## Phase success criteria

1. ≥2 promoted cycles whose search-wrapped head-to-head beats the current Kaggle
   submission's net with CI clearing 0.50.
2. The turn-boundary value-jump statistic measurably shrinks across cycles.
3. A shipped `policy.pt` passing the full phase-4 packaging + smoke + gate pipeline.
4. Total spend ≤ $2k.

## Risks / counters

- Overly sharp π at ~96 sims → temperature knob + per-cycle entropy monitoring.
- Distribution shift (search-play states vs raw-policy states) → replay-buffer mixing,
  KL warmup, anchor gates.
- Self-play meta drift → frozen-anchor + cross-deck gates every cycle.
- Engine RNG noise in gates → fixed n≥300 + Wilson CIs (established practice).
- Rental-box mortality → idempotent cycles + artifact sync.
- Gate failures → one-variable-at-a-time diagnosis loop; cheap retries.

## Out of scope

PSRO/exploiter populations; GPU inference server (except as stage-2 contingency);
search-algorithm changes; any change to the validated phase-4 agent, search, or
packaging beyond swapping `policy.pt`; B200-class hardware unless stage 2 triggers
AND distillation demands it.

## Process constraints

Commits on `main`, short lowercase messages, no co-author lines; progress appended to
`.superpowers/sdd/progress.md`; never modify `pokemon-tcg-ai-battle/`; tests CPU-only,
one pytest invocation at a time, one battle per process; cloud runs detached.
