# Competition submission — phase-3 generalist agent (design)

- Date: 2026-07-08
- Status: approved in brainstorming
- Scope: package the trained phase-3 generalist policy (`champ/phase3-generalist-r120.pt`,
  = `runs/phase3-a/checkpoint-0120`) into a Kaggle-ready submission directory for the
  Pokémon-TCG-AI-Battle competition. Bring-your-own-deck: the agent returns a fixed
  60-card deck at deck selection, then plays the game with the policy.

## Submission contract (fixed by the competition)

The competition runtime loads the submission dir at `/kaggle_simulations/agent/` and calls
`agent(obs_dict: dict) -> list[int]` once per decision (per the sample `main.py`):
- `obs.select is None` → the initial deck-selection decision → return the 60 card IDs.
- else → return a list of option indices, each in `[0, len(obs.select.option))`, length in
  `[obs.select.minCount, obs.select.maxCount]`, no duplicates.
`obs.logs` are events **since the last selection** (per-selection deltas); `obs.current.yourIndex`
is the acting seat. The agent may keep module-level state across calls within a process.

## Runtime assumptions (confirmed with user)

- Standard Kaggle image: `torch` + `numpy` available. Bundle weights + code + `cg/` only.
- In-game move choice: **stochastic sampling** (`sample_select`) — the exact inference path all
  eval win-rates were measured with (0.886 vs SD-champ on non-sample decks, ≥0.90 vs random/deck).

## Bundle layout (`submission/`, self-contained)

```
submission/
  main.py            # agent(obs_dict) entrypoint + lazy init
  deck.csv           # chosen deck (60 IDs, one per line) — set after the round-robin
  policy.pt          # policy-only state_dict extracted from checkpoint-0120 (~52 MB)
  cg/                # copied VERBATIM from pokemon-tcg-ai-battle/sample_submission/sample_submission/cg
  ptcg/              # trimmed inference package (see below)
```

- `ptcg/` contains only the inference-path modules and their closure:
  `__init__.py, cards.py, engine.py, tracker.py, featurize.py, model.py, action.py`
  (import graph verified closed over this set: featurize→cards,tracker; model→featurize,cards;
  action→model; cards→engine; engine→cg). No training/eval modules (rollout, ppo, actors,
  trainloop, replay, decks, decklists) are bundled.
- `EN_Card_Data.csv` is NOT bundled — `build_tables()` reads card metadata from the native
  `cg` module (`api.all_card_data()` / `api.all_attack()`), which `cg/` provides.

## The agent (`main.py`)

Lazy one-time init (module globals, built on first call, reused across the game):
1. `os.environ.setdefault("PTCG_ENGINE_DIR", <this dir>)` so `engine.engine_dir()` resolves to
   the bundled `cg/` (its default computes the training-repo layout, absent in the submission env);
   `sys.path.insert(0, <this dir>)` so `import cg` / `import ptcg` resolve.
2. `tables = build_tables()`; `model = PolicyModel(student_config(tables))`;
   `model.load_state_dict(torch.load("policy.pt", map_location="cpu"))`; `model.eval()`.
   `student_config` must match the config `checkpoint-0120` was trained with (student size).
3. A single `torch.Generator` (fixed seed) for sampling; the deck read from `deck.csv`.

`agent(obs_dict)`:
- `obs_dict.get("select") is None` → new game: reset per-game state (`tracker=None`, `me=None`);
  return `list(deck)`.
- else: `me = obs_dict["current"]["yourIndex"]`; if the tracker is unset or the seat changed,
  `tracker = BeliefTracker(me)`; `tracker.update(obs_dict.get("logs", []))`;
  `ts = featurize_state(obs_dict, me, deck, tracker.snapshot(), tables)`;
  `es = encode_select(obs_dict, ts, tables)`; `with torch.no_grad(): d = sample_select(model, ts, es, gen)`;
  return `d.picks`.
- Resetting on `select is None` handles a process reused across multiple games/episodes.
- This flow is byte-identical to training/eval inference, so the measured playing strength carries over.

## Robustness

The inference body is wrapped in `try/except Exception`. On ANY error it falls back to a legal
random pick: `random.sample(range(len(obs["select"]["option"])), obs["select"]["maxCount"])`
(`maxCount ≤ len(option)` per the contract). Rationale: a raised exception or crash forfeits the
game, so the agent must always return a legal selection even on an unexpected obs.

## Deck selection

`deck_roundrobin.py`: for each portfolio deck as "my deck", play the generalist deck-vs-deck against
every other portfolio deck (seats alternated, ~16 games/pair), tally each deck's average win rate
across the field, and pick the max. The chosen deck's IDs become `submission/deck.csv`. Swappable
later; the cross-deck matrix is also useful for a future deck-selection pass.

## Testing / acceptance

A local harness drives one seat of a real `BattleSession` game with `agent()` (opponent = random,
then a few games vs the generalist-on-another-deck), asserting:
- every `agent()` return is a legal selection (indices in range, count within [min,max], no dupes);
- games run to completion with no exception escaping `agent()`;
- per-move latency is within a sane budget (report the p50/p95 — flag if a move risks the clock);
- win rate vs random is sane (≫0.5), consistent with the eval numbers.
This proves obs-schema compatibility (submission `obs_dict` vs training obs dict) and legality
end-to-end before upload.

## Constraints / caveats

- Relies on the Kaggle image's `torch` accepting these weights (near-certain; standard transformer).
  If the image is bare, distillation to a numpy forward pass would be required — OUT OF SCOPE.
- Per-move latency: `sample_select` is autoregressive over `maxCount`; measured in the local test.
  If a move risks the per-step clock, greedy/argmax or batching is a follow-up.
- Never modify anything under `pokemon-tcg-ai-battle/`; `cg/` is copied, not edited.

## Out of scope (later)

Greedy-vs-sample A/B; clock management; int8/CPU distillation; multi-deck submission / deck
optimization beyond the single round-robin pick; automated Kaggle upload (produce the package;
the user uploads).
