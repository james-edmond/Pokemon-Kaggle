# PTCG AI Battle — training pipeline phase 2 (core PPO loop) design

- Date: 2026-07-04
- Status: approved in design discussion; pending spec review
- Scope: the core self-play PPO training loop that proves learning on one deck,
  running on the local desktop (CPU actors + GTX 1060 learner). Decomposition
  agreed with the user: league play and deck portfolio are phase 3; distillation,
  int8 student, IS-MCTS search, clock management, and Kaggle submission packaging
  are phase 4. This spec builds on the phase-1 architecture spec
  (`2026-07-02-model-architecture-design.md`) and must preserve its contracts.

## Constraints that shape the design

- Hardware: Intel i5-6600K (4 cores / 4 threads), 16 GiB RAM, GTX 1060 6 GB
  (Pascal, sm_61), Windows 10. Not a daily driver — multi-day runs are fine.
- Pascal reality: fp16 is crippled on GP106 (no tensor cores) — the learner
  trains fp32, no mixed precision. Modern torch CUDA wheels have dropped or
  destabilized sm_61 support, and torch builds old enough to ship sm_61 kernels
  have no CPython 3.14 wheels. Training therefore uses a dedicated venv
  (Python 3.12, torch cu121-era build); phase-1 code continues to run on the
  existing 3.14 + torch-CPU install. A plan-level spike verifies the 1060
  end-to-end (device capability (6,1), matmul + backward, full phase-1 test
  suite on the older torch). Documented fallback: learner on CPU — `device` is
  config; nothing else changes.
- Engine: one battle per process; RNG not seedable (eval is statistics with
  sized samples, never exact assertions). Measured: env step ~297 µs,
  featurize+encode ~240 µs, ~50–56 selections/game.
- Binding phase-1 contracts: the pick loop is shared actor/learner code and the
  learner replays stored pick order, never re-samples; dropout stays 0.0; the
  enumerated option list is the only legality oracle (no masking beyond the pick
  loop's picked/done rules); featurizers are pure functions; trunk never sees
  select context; nothing under `pokemon-tcg-ai-battle/` is modified.
- On any conflict with the phase-1 spec, that spec wins unless this spec
  explicitly refines it (the ratio-contract refinement below is such a
  refinement, agreed in design discussion).

## Overview

Synchronous round-based self-play PPO (approach A of the design discussion).
A run is a sequence of numbered rounds. Round N: three actor processes play a
fixed batch of mirror self-play games on CPU with the frozen current checkpoint
and write featurized trajectories to disk; the learner loads the round, runs
batched PPO epochs on the GPU, saves `checkpoint-<N+1>.pt`, appends metrics,
and deletes the round's tensors. Strictly on-policy; no staleness accounting;
resumable at round boundaries (a killed run loses at most one round).

Rejected alternatives: an async actor pool (~10–20 % more throughput, but PPO
correctness then depends on staleness bounds and Windows IPC debugging);
single-process alternating collect/learn (one engine total ⇒ ~¼ throughput;
kept as fallback if multiprocessing misbehaves, not the plan).

## Round lifecycle and data

- Run directory: `runs/<run-id>/` holding `config.json`, `checkpoint-*.pt`,
  `metrics.csv`, `rounds/<N>/` (transient), and a small persistent debug sample
  of raw observations (the first game of every 10th round — for reproducing
  featurizer issues).
- Trajectory storage: the featurized tensors the PPO smoke test already
  consumes — per step: public `TokenizedState`, `EncodedSelect`, picks, actor
  logprob, privileged `TokenizedState`; per game: result, rewards, seat map,
  seeds, featurizer version (`FEATURIZER_VERSION`), checkpoint id. Stored per
  game with `torch.save`; a round is ~1 GiB and is deleted after the learner
  consumes it (on-policy data is used once). Storing tensors rather than raw
  observations keeps the replay-exactness story trivial; the debug sample keeps
  a forensic path back to raw obs.
- Round size default: 192 games (~10–11 k steps) ≈ one hour of collection at
  student size — long enough to amortize the update, short enough to lose
  little on a crash.
- Resume: on start, the orchestrator scans for the last complete checkpoint and
  continues at that round; incomplete round directories are discarded. Optimizer
  state, round index, and RNG states live in the checkpoint.

## Actors

- Three actor processes (spawn), one engine each, leaving one core for
  learner/OS. Each plays `games_per_round / 3` mirror self-play games (both
  seats sample from the same checkpoint) via the proven B==1 `sample_select`
  path on CPU.
- Each step passes `json.loads(visualize_data())` into `featurize_privileged`
  so privileged critic states carry true deck order and hands (the phase-1
  Task-6 capability, train-only, never in the policy path).
  `rollout.play_game` gains an opt-in viz pass-through for this; otherwise
  phase-1 modules are consumed, not modified.
- Seeding: per (round, actor, game) for torch sampling; engine RNG is
  unseedable and treated as environment noise.
- Deck: the known-good sample deck on both sides for all of phase 2.

## Learner

### Batched replay (`ptcg/replay.py`)

`replay_logprob_batched(model, steps) -> [B]`: one batched trunk encode over a
minibatch of steps, then the pick loop advances all selects in lockstep —
depth 1 scores every select in one `option_logits` call (it already takes a
batch dimension and per-row `picked` masks); deeper depths continue only the
rows whose stored pick sequence goes deeper (97.6 % of selects are single-pick).
Stored picks are forced exactly as `run_pick_loop(forced=...)` does — same
masks, same order, never re-sampled.

Ratio-contract refinement (of the phase-1 "exactly" wording): bit-exactness is
guaranteed within a (device, dtype, batching) configuration and tested at
atol=0 on CPU against the B==1 path; across configurations (actor CPU vs
learner GPU) float drift is tolerated within a monitored bound — every round
asserts epoch-0 `max|ratio − 1| < 1e-3` and aborts the run on violation,
because beyond rounding noise that indicates a real policy mismatch.

### PPO (`ptcg/ppo.py`) — defaults, all exposed via `TrainConfig`

- Clip 0.2; GAE λ=0.95, γ=1.0; terminal-only reward (+1/0/−1).
- Per-seat advantages from the privileged critic exactly as the phase-1 smoke
  test assembles them; advantages normalized per round; critic trained
  alongside on the same returns.
- Value loss 0.5 × MSE on the public value head; entropy bonus 0.01;
  aux losses at 0.1 each — prize-diff, opponent-decklist, opponent-hand (true
  targets are known in self-play).
- `clip_grad_norm_(1.0)` (the phase-1 deferred cold-start fix); Adam 3e-4; fp32.
- 2 epochs per round, minibatches of 512 steps, approx-KL early stop at 0.02.
- Model config: `student_config` (d224) by default; tiny for tests; teacher
  deferred to real GPU hardware in a later phase. Everything takes
  `ModelConfig`.

## Evaluation and metrics

- Every 5 rounds, the actor pool runs an eval pass with the sampling policy:
  200 games vs uniform-random and 100 games vs each of two reference
  checkpoints (5 and 15 rounds back, once they exist).
- Logged per eval: win rates with 95 % CIs, mean game length, mean entropy,
  epoch-0 ratio drift, loss components. All metrics append to
  `runs/<run-id>/metrics.csv`; `scripts/plot_run.py` renders curves.

## Success criteria (falsifiable, one ≤72 h student-size run)

1. Win rate vs random reaches ≥65 % and holds for two consecutive evals
   (200-game samples ⇒ ±7 % CI, clearly separated from the ~50 % baseline).
2. Current checkpoint beats the 15-rounds-back checkpoint at >55 % for three
   consecutive evals (monotone improvement, not a spike).
3. Zero ratio-contract aborts, zero NaN/inf aborts, and a kill+resume mid-run
   continues cleanly with contiguous metrics.

Missing any criterion means phase 2 is not done — debug, don't reinterpret.

## Testing

All CPU, tiny config, one engine per process: unit tests for GAE/advantage
assembly against hand-computed sequences; batched-vs-B==1 replay exactness
(atol=0, same device); a one-tiny-round end-to-end integration test (3–4 games,
one actor process, real learner update, checkpoint written, resume from it);
metrics-append/resume test. The multi-day run is the plan's final verification
step, reported against the success criteria.

## Module layout

`ptcg/replay.py` (batched replay), `ptcg/ppo.py` (losses/GAE),
`ptcg/actors.py` (actor process + pool), `ptcg/trainloop.py` (orchestrator,
`TrainConfig`, checkpoint/resume), `scripts/train.py`, `scripts/plot_run.py`,
`tests/test_replay.py`, `tests/test_ppo.py`, `tests/test_trainloop.py`.

## Explicitly rejected

- Async actor pool and single-process loop (above).
- Mixed precision on Pascal (fp16 throughput penalty, no tensor cores).
- Storing raw observations and re-featurizing in the learner as the primary
  format — smaller on disk, but rounds are consumed once and deleted, and
  stored tensors keep replay exactness trivial.
- Batched acting (actors batching model forwards across parallel games) —
  a later optimization; phase 2 uses the proven B==1 acting path.

## Out of scope (→ later phase specs)

League structure and opponent sampling beyond past-self evals; deck portfolio;
reward shaping beyond terminal outcome; distillation schedule and int8 export;
search (IS-MCTS, determinization, clock); submission packaging.
