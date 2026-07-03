# PTCG AI Battle — model architecture design

- Date: 2026-07-02
- Status: approved in design discussion; pending spec review
- Scope: the neural network architecture for the battle agent — tokenization, trunk,
  heads, action decoding, teacher/student split, and the contracts training code must
  honor. The training pipeline (league design, PPO hyperparameters, distillation
  schedule, search algorithm, deck portfolio) is a separate future spec.

## Constraints that shape the design

- The engine presents every decision as an enumerated legal-option list
  (`obs.select.option`): choose k option indices, k ∈ [`minCount`, `maxCount`].
  Measured (benchmarks/RESULTS-local.md): option lists mean 6.4 / p95 17 / max ~60;
  97.6 % of selections have `maxCount == 1`; multi-pick max observed k = 3;
  ~28 selections per player per game.
- Partial information: opponent hand/deck/prizes are hidden; your own deck and prizes
  are known as a union but not as a partition; reveals arrive through the `logs`
  stream, not the state snapshot.
- Submission runs CPU-only (no GPU guarantee) under a 10-minute-per-player chess
  clock — ~9–10 s per non-forced move. The engine's forward model costs a measured
  0.42 ms per `search_step` (including dataclass decode; a leaner decode should
  land near 0.3 ms), so search-based inference is planned and the network must be
  cheap on a single CPU core.
- `cg/api.py` warns in three places that enum members and dataclass attributes may
  be appended mid-competition. Every fixed-vocabulary surface must degrade
  gracefully on unseen members.

## Overview

An entity-transformer encoder over the public game state (a pure state encoder —
no select-context input), plus a shallow per-select decoder with a flat pointer
head that scores the enumerated options. A privileged critic runs as a separate
full-information encoder at training time and is optionally reusable at
determinized-search leaves. Deck conditioning falls out of the own-list multiset
tokens; no separate mechanism. A large teacher trains on GPU; a small int8 student
is distilled for Kaggle CPU inference.

Because the trunk never sees the select context, one trunk pass per state prices
every option at that state: the flat head scores the whole enumerated list in a
single forward. This is a load-bearing property for search (one encode per node)
and for any future caching.

## Tokenization

Owned by the env wrapper: a featurizer plus belief tracker implemented as a pure
function of the observation stream, imported by actor, learner, and search alike —
single source of truth, versioned, version logged into trajectories.

### Entity tokens — one per pointable, stateful thing

- Pokémon in play, both players (active + bench): hp/maxHp, damage, special
  conditions (active only, per rules), `appearThisTurn`, energy summary by type,
  stage/ex/tera flags.
- Child tokens, parent-linked, for every attached energy card, tool, and
  pre-evolution card. Required for pointability: options reference attachments
  directly (`OptionType.ENERGY_CARD`/`TOOL_CARD`/`ENERGY` with
  `energyIndex`/`toolIndex`; contexts like `DISCARD_ENERGY_CARD`).
- Stadium; each card in own hand; `looking` cards when present.

### Multiset tokens — one per distinct card id, with a remaining-count feature

- Own deck ∪ prizes (the union is known; the partition is not — split sizes enter
  as scalar features).
- Own discard; opponent discard; opponent revealed cards (belief tracker output:
  cards revealed to their hand or deck via log replay; shuffle events demote
  known-position to known-membership).
- Face-up prize cards, if any.
- Copies of the same id are fungible, so options that reference a card in one of
  these zones (e.g. discard-to-hand selects, `select.deck` searches) resolve to the
  distinct-id token for that card.

### Scalar features for identity-unknown quantities

Opponent hand count, both deck counts, prize counts, own deck/prize split sizes.

### Special tokens

Two player-summary tokens, one global token (turn, `turnActionCount`,
`firstPlayer`, per-turn flags, zone counts, remaining-clock feature), two value
readout tokens, four scratch registers. There is no decision-context token — the
select context is decoder-query-side (below). Plan tokens are deferred (see
Deferred).

### Per-token composition

Card-identity encoding ⊕ numeric features ⊕ segment embeddings (owner:
self/opponent/neutral; zone; token kind: entity/child/multiset/special; stack
position: child index, bench index).

Card identity = learned embedding table (2,102 ids + reserved rows: PAD, a
hash-bucket UNK, headroom for mid-competition additions) ⊕ printed-attribute
features from `all_card_data()` (HP, types, stage, attack costs/damage, retreat,
ex/megaEx/tera/aceSpec flags) ⊕ optional initialization from a frozen text
encoding of the card's rules text, projected into the table.

Typical sequence length ~100–150 tokens; hard cap with bucketed padding lengths
for batch efficiency (bucket sizes are an implementation choice).

## Trunk

Pre-LN transformer encoder. Teacher: d = 512, 8 layers, 8 heads, MLP ratio 4 —
~27 M parameters including embeddings. Input is the token set above and nothing
else: no select information reaches the trunk.

## Per-select decoder (policy)

- Query: fused select features — `SelectType` and `SelectContext` embeddings
  (with hash-bucket fallback rows), `minCount`/`maxCount`, `remainEnergyCost`,
  `remainDamageCounter`, and the `contextCard`/`effect` references resolved to
  their entities' trunk outputs.
- Option encoding is compositional over the option struct's fields:
  `OptionType` embedding + resolved entity references (area/playerIndex/index →
  that token's trunk output; `attackId` → attack embedding; `cardId` → card
  identity encoding; `number`/`count` → scalar embedding). Compositional
  featurization is a robustness requirement: an enum member appended
  mid-competition lands somewhere sane instead of keying an OOV crash.
- Two cross-attention decoder layers (options + query attending into trunk
  outputs) in the teacher — the student drops to one if latency demands — then an
  MLP, one logit per option, softmax over the list. No masking anywhere:
  the enumerated list is the legality oracle.
- Multi-pick and optional selections run autoregressively: sample an option, add
  its value vector into the decision stream (query update), re-score the
  remainder; a "done" action is available once picks ≥ `minCount` and maps to
  emitting the selection list. The joint log-probability is the sum over picks and
  constitutes the select's single composite-action log-prob — valid because all
  picks issue from one state with no environment information arriving between
  them.
- Contract: the pick loop is shared actor/learner code. The learner replays the
  stored pick order deterministically to recompute the joint log-prob; it never
  re-samples or re-derives.

## Value heads and auxiliaries

- Public value head (policy trunk, value tokens): scalar E[outcome] ∈ [−1, 1]
  with win/draw/loss = +1/0/−1, plus a prize-differential auxiliary target for
  denser signal. Ships in the student (search-leaf fallback, time management).
- Privileged critic: a separate encoder instance of the same family (may be
  narrower) over the full-information tokenization — true opponent hand, true
  deck/prize partitions, available in self-play. Outputs per-player values. Used
  for GAE at training time; optionally reusable at inference on
  determinized-search leaves, where the sampled world is fully specified. Never
  ships in the policy path.
- Train-only auxiliary heads on the policy trunk: opponent decklist multiset,
  opponent hand distribution, opponent archetype cluster. The archetype head can
  seed determinization sampling at inference.

## RL interface contracts (details belong to the training spec)

- One PPO timestep = one `battle_select`: its own stored log-prob, value
  bootstrap, GAE advantage, and clipped ratio. Within-select picks form one
  composite action as above. Whole-turn composite actions are rejected: clipping
  a product over a turn's selects is coarse (one drifted component saturates the
  clip or hides behind an offsetting drift), and a single turn-level advantage
  smears credit that per-select GAE localizes.
- γ ≈ 1, terminal-only reward as the baseline; shaping decisions deferred to the
  training spec. The per-select vs per-turn γ-horizon difference is noise at
  these settings; the requirement is consistency.

## Teacher → student

- Student: same architecture family, target d ≈ 224, 4 layers (~3 M parameters
  plus embeddings), int8 (ONNX Runtime), targeting ≤ 2 ms per evaluation on one
  Kaggle CPU core. With `search_step` measured at 0.42 ms, that is ~2.4 ms per
  node before per-node featurization and tree overhead — budgeting conservatively,
  roughly 1.5–2.5k search nodes per pivotal move inside ~9 s.
- Distillation target = the teacher's policy over the enumerated options + the
  public value. Nothing forces token-for-token input parity, so the student may
  compress its input representation (pooled multisets, capped child tokens) if
  CPU latency demands.
- Sizing protocol: compare candidate students at equal wall-clock strength with
  search in the loop — not at equal node counts — because whether better priors
  dominate the node deficit is exactly the empirical question.

## Deferred, behind flags, with enablement bars

1. Trunk caching across within-effect selects. Enable only if profiling shows the
   acting loop is encoder-bound. Required shape: the encode boundary is a pure
   function of the observation stream living in the shared env wrapper; boundary
   decisions are logged into trajectories; the learner replays and asserts, never
   re-derives. Expected win ≤ ~2× and acting-loop-only (unmeasured); exactly zero
   effect in search, where every node is a distinct hypothetical state.
2. Plan tokens (within-turn recurrence). Bar to enable: must beat the query-side
   decision-context features in ablation; carried state enters search node
   identity (hash it into the transposition key or forgo transpositions); the
   recurrent state must be stored and replayed through the learner.

## Explicitly rejected

- Verb-head factorization with derived legality masks — redundant against the
  enumerated option list (the only legality oracle), fragile to mid-competition
  enum appends, and unmotivated at ≤ ~60 options; type-as-feature already provides
  the parameter sharing that would be its defense.
- Recurrent trunk / full history transformer — search-hostile (carried state at
  every hypothetical node) and largely redundant with the belief tracker, which
  recovers the countable hidden information losslessly. Revisit only on a
  demonstrated memory gap the tracker cannot close.
- Flat-vector MLP (discards entity structure and option tying); GBDT as the
  policy learner (wrong function class for policy-gradient self-play).

## Open questions deliberately out of scope (→ training-pipeline spec)

League structure and opponent sampling; PPO hyperparameters; reward shaping;
distillation schedule; search algorithm specifics (IS-MCTS variant,
determinization sampling from the aux heads, clock management); deck portfolio
selection; featurizer throughput budget against the measured 297 µs/step env
cost.
