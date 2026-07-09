# Inference-time search (IS-MCTS) — phase 4 (design)

- Date: 2026-07-09
- Status: approved in brainstorming
- Scope: add inference-time search to the Kaggle submission agent to strengthen play of the
  frozen phase-3 generalist (`champ/phase3-generalist-r120.pt`, extracted to
  `submission_src/policy.pt`) — no retraining. Architecture: PUCT MCTS over K determinized
  engine-search trees with a root visit-count vote ("approach B"), aux-head-guided
  determinization, public-value leaf evaluation, and wall-clock budget management.
- Builds on: `2026-07-08-competition-submission-design.md` (the validated agent/packaging),
  phase-3 spec (IS-MCTS listed there as out-of-scope/deferred).

## Context and pre-verified facts

The self-play policy is converged (phase3-a plateau; more training does not help). Search at
inference is the top remaining strength lever. The user confirmed the current submission
(87f1b21: native cg loaded in-agent) **validates on Kaggle**, so in-agent engine search is
viable.

Live probes (2026-07-09, this machine, mid-game positions on the sample-deck mirror) —
the design rests on these empirical facts:

- `search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active,
  manual_coin)` builds a simulable root **from the agent's obs** (`search_begin_input`
  string) in ~1 ms. Works at MAIN and non-MAIN selects. Root obs mirrors the live select
  (same options), `search_begin_input=None` inside search.
- `search_step(search_id, picks)` ≈ 0.26–0.34 ms including JSON parse; returns a **fresh
  searchId per call** (stepping one parent twice with the same picks gives two independent
  children — chance re-sampled). States persist until `search_release`/`search_end`;
  multiple `search_begin` roots coexist; `search_end()` frees the arena for reuse.
- **Perspective flips to the acting player** during search: at opponent decisions,
  `current.yourIndex` = opponent, their hand is visible, ours hidden; `obs.logs` are
  per-viewer deltas delivered at that seat's next decision (same semantics as live play).
- **Hidden-zone order is engine-shuffled**: identical `search_begin` inputs + identical
  picks produced different draws across two begins. We control the *multiset split*
  (deck vs prize vs hand contents), never ordering. Supplying orderings is pointless.
- Terminal states surface `current.result` (0/1 winner, 2 draw); stepping past terminal
  raises ValueError (error 3) — catchable.
- Coins: 3,200+ random search steps of the sample-deck mirror produced **zero** COIN logs
  and zero COIN_HEAD selects under both `manual_coin` values → coins are treated as
  environment randomness (`manual_coin=False`); explicit coin chance-nodes are out of scope.
- Select census (search random walks): ~69% MAIN; **98% of selects have maxCount==1**;
  rest ≤3. → priors need one `option_logits` pass at almost every node.
- NN cost (this desktop, torch CPU, student d224): featurize+encode 0.16 ms; trunk encode
  11.2 ms (B=1), 6.6 ms/state (B=4); one `option_logits` pass 3.6 ms; value/aux heads
  ~0.4 ms off a shared trunk; full `sample_select` 12.3 ms. Engine steps are ~300× cheaper
  than NN evals → AlphaZero-style search (priors + value leaves, no NN rollouts).
- Raw-JSON integration: `lib.SearchBegin/SearchStep` return JSON whose
  `state.observation` dict has exactly the live obs schema (`select/logs/current`) → the
  existing featurizer consumes search obs dicts directly.

Trained heads available on the frozen policy (ptcg/model.py, trained in ppo/trainloop):
`public_value` (Tanh, MSE vs ±1/0 returns, acting-seat perspective — the leaf evaluator),
`prize_diff` (unused by this design), `aux_decklist` / `aux_hand` (softplus per-card-row
expected counts, Poisson-trained vs the true opponent decklist/hand — the determinization
proposal).

## Kaggle runtime constraints (hard)

- Agent loaded via `exec(code, env)` with **no `__file__`** (main.py `_agent_dir()` handles).
- `actTimeout=0` (no per-move limit) but a **600 s overage bank per agent** and ~2000 s
  episode cap: every second of thinking drains the bank → wall-clock budgeting required.
- Python 3.11, torch + numpy available, 12.2 GB memory, CPU only.
- The agent must never raise: search → raw policy → legal-random fallback ladder.
- Never modify anything under `pokemon-tcg-ai-battle/`.

## Architecture

Four new pure-inference modules in `ptcg/` (bundled into the submission alongside the
existing seven):

- **`ptcg/simsearch.py`** — ctypes wrapper owning its own `agent_ptr = lib.AgentStart()`;
  calls `lib.SearchBegin/SearchStep/SearchRelease/SearchEnd` directly and returns obs
  **dicts** via `json.loads` (no dataclass layer). Replicates `cg.api.search_begin`'s
  argument validation (count checks, `select.deck != None` → `your_deck=[]`, facedown-active
  handling). Depends only on `cg.sim.lib` symbols. API: `SearchSession.begin(obs_dict,
  det) -> (sid, obs) | None`, `.step(sid, picks) -> (sid, obs) | None` (None on engine
  error, never raises), `.end()`.
- **`ptcg/determinize.py`** — `sample_determinization(obs, me, my_deck, belief, dl_lambda,
  hd_lambda, tables, rng) -> Determinization` (the six card-ID lists for `search_begin`).
  Never raises; see Determinizer below.
- **`ptcg/mcts.py`** — K-tree PUCT search; see Search below. Entry point:
  `search_move(obs_dict, ctx) -> list[int] | None` where ctx carries model/tables/trackers/
  clock/rng (None → caller falls back to raw policy).
- **`ptcg/clock.py`** — pure time-bank logic, no engine/torch imports; see Clock below.

Per-move data flow in `submission_src/main.py`:

1. tracker update (existing, unchanged);
2. trivial-select fast path: exactly one legal pick-list → answer immediately (no NN);
3. featurize root once → one trunk encode → root priors + `aux_decklist`/`aux_hand` λs off
   that single trunk (no extra NN cost);
4. sample K determinizations → K × `search_begin`;
5. PUCT simulations round-robin across trees until the move's time slice expires;
6. root vote: sum visit counts per canonical pick-list across trees (root action space is
   identical in all trees — same real select); argmax, ties by summed mean value;
7. `search_end()` in `finally`; any failure anywhere → existing raw-policy path.

## Determinizer

Output invariant: exact zone counts (`len(your_deck) == my deckCount`, etc. — engine
validates), all real card IDs.

- **My side**: unseen multiset = my 60-card decklist − my hand − my visible zones
  (discard, in-play incl. attachments/pre-evolutions, revealed own prizes). Split uniformly
  at random into unrevealed-prize slots vs deck (prizes are uniform in reality; order is
  engine-shuffled anyway). Revealed own-prize identities placed exactly.
- **Opponent side**, merging three layers:
  1. *Visible* opponent cards (discard, board incl. attachments/pre-evolutions, revealed
     prizes, their stadium) are fixed and excluded from sampling.
  2. *Tracker constraints* (`BeliefSnapshot`): `opp_hand` counts are hard minimums in hand,
     `opp_deck` minimums in deck, `opp_hidden_pool` must land in hand∪deck.
  3. *Aux guidance*: hidden-card weights = `clamp(dl_lambda − visible_counts, ≥0)`;
     sample the remaining unknown cards by weighted draw without replacement, respecting a
     4-per-name cap and ≤1 ACE SPEC (basic energy uncapped; realism, not engine rules).
     Zone assignment: hand = known-hand minimums + unknowns weighted by `hd_lambda`;
     deck = known-deck minimums + remainder; prizes uniform from what's left after hand.
- Card-row→ID via the tables' inverse map; unmappable rows → filler = a basic energy of
  the opponent's observed energy type (else any common basic energy ID).
- Facedown opponent active (setup): `opp_active` = λ-most-likely **Basic Pokémon**
  (engine error 2 requires a Pokémon ID). Setup rule "opponent deck must contain ≥1 Basic"
  force-satisfied by swapping in a filler basic if the sample lacks one.
- Contract: never raises. If `search_begin` rejects a sample (error 1/2), retry that tree
  once with a pure-filler determinization; on second failure drop the tree (K shrinks;
  K=0 → raw-policy move).

## Search (per tree: PUCT; across trees: root vote)

- **Node**: engine `searchId`, obs dict, acting seat, terminal flag/value, proposed actions
  (pick-lists), priors P, visit counts N, total values W, children, and two cloned
  `BeliefTracker`s (mine + opponent's). On expansion, the child's obs logs update the
  **child's acting seat's** tracker clone (probe: logs are per-viewer deltas at that seat's
  decisions); the other tracker passes through unchanged.
- **Action proposal**: single-pick selects (98%) → every option index, priors = softmax of
  one `option_logits` pass (done column masked when minCount≥1; `[]` included as an action
  when minCount==0, prior from the done column). Multi-pick selects → M=8 autoregressive
  policy samples + the greedy pick-list, deduped; priors ∝ sampled probabilities,
  renormalized.
- **Selection**: PUCT, `Q(a) + c_puct · P(a) · √(ΣN) / (1 + N(a))`, `c_puct = 1.5`
  (tunable). Q stored from the node's acting seat; negamax sign flip on backup across seat
  changes (value at any node = expected result for that node's acting seat).
- **Expansion + leaf eval**: `search_step` (~0.3 ms) → child obs → featurize from the
  child's acting seat (`me` = child `yourIndex`; deck = my real decklist for my seat, the
  tree's determinized opponent decklist for theirs; belief = that seat's tracker snapshot)
  → one trunk encode → store priors + `public_value` (sign-adjusted to the root seat).
  Terminals: exact +1/0/−1 from `result`. The opponent is thus played by the same policy
  with the same information structure it trained on (empty opponent tracker at root —
  approximation, documented).
- **Chance**: implicit — each expansion samples one stochastic outcome (draw/shuffle/coin)
  and caches the child (standard determinized-UCT bias, mitigated by K independent trees
  and the engine re-sampling per tree).
- **Root decision**: argmax over canonical pick-list keys of ΣN across trees, ties by
  ΣW/ΣN; deterministic (no temperature, no Dirichlet).
- Defaults (runtime-tunable constants): K=6 trees, ≤64 simulations/tree, leaf eval B=1
  (batched B=4 collate is a measured 1.7× — deferred optimization).

## Clock management (`ptcg/clock.py`)

All budgets in **measured wall time** (self-calibrates to Kaggle CPU; no eval-count
assumptions). Per game: search bank = 480 s (600 s overage − 120 s safety margin; model
load and non-search moves also drain the real bank, hence the margin).

- Per move: `slice = clamp(remaining_bank / expected_remaining_moves × importance, 0, 20 s)`.
- `expected_remaining_moves = max(20, 80 − own_moves_made)` (census: ~40–80 own selects).
- Importance: MAIN selects containing an ATTACK option or with ≥6 options → 1.5×; all
  other non-trivial selects → 1.0×.
- Trivial selects — exactly one legal pick-list (`nopt==1 && minCount≥1`, or forced
  take-all `minCount==maxCount==nopt`) → slice 0, answered instantly, no NN call.
  (`nopt==1 && minCount==0` is a real choice — [] vs [0] — and is searched.)
- Floor: remaining bank < 60 s → search disabled for the rest of the game (raw policy,
  ~12 ms/move).
- The PUCT loop checks the slice between simulations and stops on expiry; `search_end()`
  runs in `finally`. Time spent is measured per move and decremented from the bank.

## Agent integration (`submission_src/main.py`)

- Search state initializes lazily alongside the model (AgentStart + tiny self-check) behind
  a `_SEARCH_OK` flag; failure → flag off, agent identical to the currently-validated build.
- Per-move guards: `obs_dict.get("search_begin_input")` must be a non-empty str, clock must
  allow, `_SEARCH_OK` true — else raw policy.
- The entire search path is wrapped in try/except → raw-policy path → `_is_legal` /
  `_fallback` (existing ladder unchanged). Search results are validated with the same
  `_is_legal` before returning.
- Seeding: torch Generator stays seed 0; search RNG = `random.Random(hash((game_no,
  move_no)))`-style fixed scheme for reproducibility.
- Telemetry to stderr: per-game counters (moves searched, sims run, fallbacks, time spent) —
  visible in Kaggle agent logs, no contract change.
- Per-game reset on `select is None` also resets clock/counters (existing reset hook).

## Packaging

- `scripts/make_submission.py`: bundle the four new modules in the `ptcg/` subset; keep
  `cg/` (native lib) as today. `--no-search` flag builds a rollback zip whose `main.py`
  has search disabled (bit-identical agent behavior to the validated build).
- Self-containment check additionally asserts the bundle can `lib.AgentStart()` (full
  search needs a live obs and is exercised by the game smoke instead).
- `dist/`, `policy.pt` remain gitignored.

## Testing / acceptance

CPU-only, one battle per process, one pytest invocation at a time (existing rules).

- **Unit**: determinizer — exact zone counts, constraints honored (known-hand ⊆ hand etc.),
  50 random mid-game samples all accepted by `search_begin`, never-raises property, filler
  fallback; simsearch — begin/step/end round-trip on a live obs, engine errors → None (not
  raise), post-terminal step → None; mcts — PUCT math on a fake engine/net (negamax signs,
  prior masking/renorm, root-vote aggregation, [] action handling); clock — slice math,
  importance, floor, trivial-select detection.
- **Integration**: `scripts/test_submission.py` extended — full games with search forced on
  at a small budget: every return legal, games complete, telemetry shows searches ran,
  per-move latency within slice + overhead bound.
- **Ship gate (user-agreed)**: `scripts/eval_search.py` — search agent vs raw-policy agent,
  sample-deck mirror, alternating seats, reduced budget (~0.5–1 s/move, K=3), a few hundred
  games via the actor-pool pattern: require **win rate ≥ 0.55 with the 95% CI above 0.50**;
  plus ~20 full-budget games (sanity + latency histogram) and a small non-mirror check on
  2–3 portfolio decks vs raw policy. No legality/crash/timeout regressions.
- Then rebuild `dist/submission.zip`, re-run packaging checks, hand to the user for Kaggle
  upload (user submits; `--no-search` zip is the rollback).

## Out of scope (future levers)

Batched leaf evaluation (B=4 collate, ~1.7×); explicit coin chance nodes
(`manual_coin=True`); single-tree IS-MCTS with canonical action keys; root exploration
noise/temperature; opponent-model tracker warm-start from public history; distillation /
int8; deck re-selection under search (round-robin re-run with the search agent).

## Process constraints

Commits on `main`, short lowercase messages, no co-author lines; progress appended to
`.superpowers/sdd/progress.md`; training venv only for GPU work (none here); never modify
`pokemon-tcg-ai-battle/`.
