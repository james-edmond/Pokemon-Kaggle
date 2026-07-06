# PTCG AI Battle — league play + deck portfolio (phase 3) design

- Date: 2026-07-05
- Status: approved in design discussion; pending spec review
- Scope: multi-deck, population-based (league) self-play training that makes the
  policy robust across a diverse field of decks and opponents — the actual
  bring-your-own-deck competition setting. Follows the phase-2 training loop
  (`2026-07-04-training-pipeline-phase2-design.md`) and reuses its numerical
  machinery unchanged. Distillation, inference-time search (IS-MCTS), clock
  management, and submission packaging remain phase 4.

## Motivating findings (phase-2 diagnostic, 2026-07-05)

- The phase-2 single-deck mirror policy is **solved/converged**, not buggy. A
  round-robin over checkpoints 0..30 (100 games/pair) showed fast transitive
  improvement rounds 0→10 (5 beats 0 at 82 %, 10 beats 5 at 83 %, win-rate vs
  random 38 %→89 %), then a hard plateau: all post-round-10 checkpoints are
  mutual coin-flips (adjacent matchups 46–54 %), vs-random pinned ~89 %, only
  weak non-transitivity (3 cycles, edges 0.51–0.56). Entropy stabilized ~0.95
  (not collapsed; p90 ≈ 1.9) and the value head **improved** (outcome
  correlation 0.46→0.60, monotone calibration) — ruling out entropy collapse
  and critic failure. The mirror match simply ran out of gradient.
- The competition is **bring-your-own-deck**: the submission agent returns a
  60-card deck on the first decision (`obs.select is None`), and opponents bring
  their own. The card pool is real, current Standard-format Pokémon TCG (SV-era
  expansions: TWM, SSP, TEF, PRE, …; ~2022 cards). Single-deck mirror training
  therefore optimizes a problem the competition never poses and leaves ~2000
  cards with untrained embeddings and zero non-mirror matchup experience.

Conclusion: the lever is **deck and opponent diversity**, delivered by a league
over a portfolio of real metagame decks. No architecture change is required —
phase 1 made deck-conditioning fall out of the own-deck multiset tokens.

## Constraints that shape the design

- Reuse all phase-2 contracts verbatim: shared pick loop / batched replay with
  the ratio-one contract (epoch-0 `max|ratio−1| < 1e-3` abort), dropout 0.0,
  option list as the only legality oracle, pure featurizers, one battle per
  process, fp32 on the GTX 1060 with `--minibatch 128` (512 OOMs), checkpoint
  IO/resume, models-to-device-before-optimizer-restore. Never modify anything
  under `pokemon-tcg-ai-battle/`.
- The engine is the deck legality oracle: `battle_start` rejects an illegal
  deck with an error type. Deck legality is validated by the engine, not by
  reverse-engineering the format rules.
- Long-running training must be launched fully detached (never as a child of a
  transient session) — a phase-2 lesson (teardown killed a run).
- Commit style: short lowercase messages, directly on `main`.

## Overview

Round-based synchronous self-play as in phase 2, extended so each game samples
(a) a deck for each side from a curated portfolio and (b) an opponent policy
from a mixture of {current policy, a past-snapshot league member, random}. The
learner always plays the current policy and trains on its own seat's
trajectories; mirror games (opponent = current) additionally yield the opponent
seat. A dedicated league pool of policy-only snapshots grows every few rounds
with capped, anchored retention. Evaluation gains a deck axis and an opponent
axis, with the frozen phase-2 single-deck champion as the generalization
yardstick.

Rejected alternatives: prioritized / PSRO-style opponent selection (payoff-matrix
bookkeeping — a phase-4 refinement); multi-deck without past opponents (leaves
the weak cycling unaddressed, risks re-plateauing per deck).

## Deck portfolio (`ptcg/decks.py`)

- Primary source: **curated real metagame decklists**. Current Standard
  tournament lists for the top archetypes (Charizard ex, Dragapult ex, Raging
  Bolt ex, Gardevoir ex, Miraidon ex, Lugia VSTAR, Gholdengo ex, a Lost-Zone
  toolbox, etc.), plus the phase-1 Gardevoir sample deck. Kaggle public entries
  (already card-ID `deck.csv` files) drop in directly.
- Name→ID resolver: map `Card Name` (+ `Expansion` / `Collection No.` when
  needed to disambiguate) from `pokemon-tcg-ai-battle/EN_Card_Data.csv` to a
  canonical card ID per functional card (reprints share function; pick one ID).
- Two-gate validation for every deck: (1) engine `battle_start` accepts it
  (legality); (2) a short random-vs-random run yields normal-length games with
  both win and loss outcomes (playability — mainly guards the mapping, since
  real lists are playable by construction).
- Fallback: programmatic assembly (coherent evolution lines, type-matched
  energy, enough basics) only to fill diversity gaps.
- Deliverable: `PORTFOLIO: dict[str, list[int]]` of ~8–10 validated archetypes
  (incl. the sample deck), committed as data for reproducibility, with distinct
  types and tempos (≥1 fast-aggressive, ≥1 slower evolution-heavy). All decks
  are used in training; there is no held-out-deck set in phase 3 (avoids the
  untrained-embedding problem — generalization is measured against SD-champ).

## League (`ptcg/league.py`)

- Snapshot store: policy-only weights (no optimizer state) under a `league/`
  dir, separate from resume checkpoints. Added every `snapshot_every` rounds
  (default 5, aligned with eval).
- Retention: capped pool (`pool_cap`, default ~18) with anchor retention —
  round 0 and a couple of early snapshots are never evicted (preserves
  punishment of weak play); when full, evict from the middle.
- Per-game sampling (independent draws): `my_deck`, `opp_deck` uniform from the
  portfolio (mirror deck-matchups allowed); opponent policy from a mixture —
  current (mirror) / past-snapshot / random at `mirror_frac` / `pool_frac` /
  `random_frac` (defaults ~0.30 / 0.65 / 0.05). `learner_seat` chosen per game.

## Data collection change (the only correctness-critical change)

- Mirror games (opponent = current policy): both seats are on-policy — collect
  both seats' steps, exactly as phase 2.
- Pool / random games: only the learner seat is on-policy — collect **only the
  learner seat's steps**; the opponent seat's actions came from other weights
  and are invalid for the current policy's PPO update.
- Everything downstream is unchanged and runs only over collected learner-seat
  steps: terminal reward ±1/0, per-seat GAE from the privileged critic (which
  still reads full-information state for the learner seat), batched replay, the
  epoch-0 ratio gate, KL stop, grad clipping. The ratio-one contract holds
  because opponent steps never enter replay.
- Halved data per non-mirror game is absorbed by the mirror fraction and a
  modest `games_per_round` increase.

## Evaluation and success criteria

Eval (`_eval_due` / eval_worker) gains a deck axis and an opponent axis:

- **Generalization yardstick:** freeze the phase-2 champion (`checkpoint-0031`,
  the "trained on one deck" control) as `SD-champ`. Each eval, current vs
  SD-champ across the portfolio, both sides on the same deck each game
  (seats alternated). Headline = win rate vs SD-champ averaged over the
  non-sample decks.
- **League health:** current vs recent pool snapshots (>55 % = progressing) and
  vs a frozen early anchor (stays high = not forgetting).
- **Per-deck vs random:** one number per trained deck.
- **Cross-deck matchup matrix:** periodic diagnostic; current on deck A vs deck
  B over the portfolio (archetype strength; seeds later deck selection).

All win rates are sized-sample statistics with 95 % CIs; engine RNG is
unseedable, so no exact-value assertions.

Success criteria (falsifiable):
1. **Generalization:** multi-deck policy beats frozen SD-champ >60 % averaged
   across non-sample decks, on two consecutive evals.
2. **Sustained gradient:** current beats the 15-rounds-back league snapshot
   >55 % for materially longer than the single-deck mirror sustained it (which
   stalled by round ~10–15).
3. **Per-deck competence:** ≥70 % vs random on every trained deck by the end.
4. **Stability:** zero ratio-gate aborts, zero NaN/inf, clean kill+resume with
   contiguous metrics (carried from phase 2).

## Module layout

- New: `ptcg/decks.py` (name→ID resolver, curated decklists as data, two-gate
  `validate`, `PORTFOLIO` registry); `ptcg/league.py` (policy-only snapshot
  store, mixture sampling, capped+anchored retention).
- Modified: `ptcg/rollout.py` — a league game function playing learner vs
  opponent that collects the learner seat (or both for mirror); `ptcg/actors.py`
  — round worker samples `(deck_pair, opponent, learner_seat)` per game, loads
  the opponent snapshot, writes learner-seat trajectories; eval_worker adds
  per-deck + vs-SD-champ. `ptcg/trainloop.py` — league snapshotting each N
  rounds; new `TrainConfig` fields (`mirror_frac`, `pool_frac`, `random_frac`,
  `snapshot_every`, `pool_cap`, `sd_champ_ckpt`, portfolio selector,
  `games_per_round` bump); per-deck / vs-champ metrics fields.
- Scripts: a decks build/validate script; `scripts/train.py` flags extend
  automatically from `TrainConfig`; `scripts/plot_run.py` gains the new curves.

## Explicitly out of scope (→ phase 4)

Distillation to the int8 CPU student; inference-time search (IS-MCTS,
determinization from the aux heads, clock management); Kaggle submission
packaging; deck *selection*/optimization (which deck to submit — the cross-deck
matrix seeds it, but choosing and tuning a submission deck is later);
prioritized/PSRO opponent selection; held-out-deck generalization (needs the
untrained-embedding problem addressed).
