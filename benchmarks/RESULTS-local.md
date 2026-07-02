# PTCG engine self-play throughput — local benchmark

Date: 2026-07-02. All numbers below come from commands run in this session; raw
stdout is appended chronologically to [bench-local.log](bench-local.log).

## Hardware / platform

> **Deviation from the brief:** the brief assumed a macOS laptop (`sysctl`,
> `libcg.dylib`, quarantine checks). The actual local machine is a **Windows 10
> desktop**. `cg/sim.py` auto-selects `cg.dll` on Windows, so the engine runs
> unmodified; macOS preflight steps were replaced by Windows equivalents (PE
> architecture check instead of `file`, NTFS alternate-data-stream check instead
> of `xattr`). The scripts themselves are stdlib-only and portable to Linux.

| Item | Value |
|---|---|
| CPU | Intel Core i5-6600K @ 3.50 GHz (Skylake, 2015), 4 cores / 4 threads, no SMT, no E-cores (homogeneous) |
| RAM | 16 GiB |
| OS | Windows 10 Pro 10.0.19045, 64-bit (Gigabyte desktop board; WMI misreports `PCSystemType=2` "Mobile" — the 6600K is a socketed 91 W desktop part) |
| Python | 3.14.6 (MSC v.1944, AMD64) — satisfies the 3.10+ requirement of `cg/api.py` |
| Engine lib | `cg.dll`, PE machine 0x8664 (x86-64) — matches interpreter; 1,525,248 bytes; no `Zone.Identifier` stream (no mark-of-the-web) |

This is a desktop, not a laptop: thermal throttling is unlikely and no signs of
it were seen (late runs were not slower than early ones — the opposite, see
Anomalies). Numbers are still a lower bound vs. modern server cores per-clock,
but a 3.5–3.9 GHz Skylake core is roughly in the range of sustained cloud-CPU
per-core speed.

## Methodology

- `time.perf_counter` throughout; 3 warmup games before every timed window;
  fixed per-game Python RNG seeds (see caveat below); mean/median/p95 reported.
- Sample sizes: 200 games (structure), 100 games (decomposition), ≥30 s and
  ≥30 games (throughput windows), 30 s per scaling config.
- Random policy identical to the sample agent: `rng.sample(range(len(option)), maxCount)`.
- `battle_finish()` after every game; selection cap 5,000/game — **never hit**
  in any run (max observed 219 selections).
- Multiprocessing uses spawn; each worker pays import + `GameInitialize` +
  warmup **before** a barrier; only the post-barrier window is timed.
- **Reproducibility caveat (measured, not assumed):** fixing Python seeds does
  *not* make games reproducible — the engine has its own internal RNG for
  shuffles/coins, not seeded via the public API. Two structure runs with
  identical seeds produced different game sets (10,088 vs 11,057 total
  selections; P0 wins 99 vs 108). Expect game-mix variance between runs, and
  treat "games/s" as having ~±5–9 % run-to-run noise.

Scripts (stdlib-only, paths resolved relative to repo root):
[common.py](common.py), [smoke_test.py](smoke_test.py),
[bench_structure.py](bench_structure.py), [bench_stepcost.py](bench_stepcost.py),
[bench_scaling.py](bench_scaling.py), [bench_memory.py](bench_memory.py),
[bench_search.py](bench_search.py).

## Preflight smoke test

One random-vs-random game via the documented `cg.game` API (sample deck both
sides): 81 selections, 23 turns, result 0 (P0 win), 0.025 s wall, no cap hit.
First observation is a YES_NO / IS_FIRST selection, as expected when decks are
passed to `battle_start` directly.

## 1. Game structure (200 games, single process)

From the canonical run (seeds 1000–1199, per-player split; first run's
distributional stats were consistent within sampling noise):

| Metric | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| Selections/game (both players) | 55.3 | 40.5 | 128 | 13 | 200 |
| Selections/game/player (n=400) | 27.6 | 21 | 68 | 3 | 101 |
| Selections/game, busier player | 31.0 | 24.5 | 68 | 10 | 101 |
| Turns/game | 15.6 | 12 | 38 | 3 | 65 |
| Option-list size (n=11,057 selections) | 6.35 | 5 | 17 | 1 | 50 |
| Game wall time (full `game.py` dict API) | 16.2 ms | 11.9 ms | 40.2 ms | 4.1 ms | 64.3 ms |

- `maxCount` is 1 for 97.6 % of selections (2: 74, 3: 188 of 11,057).
- **Forced selections** (`maxCount == 0` or `== len(option)`, i.e. exactly one
  legal action): **13.0 %**. Selections with ≤2 options: 25.5 %. This is well
  below the brief's "~half trivial" working assumption — budget accordingly.
- Results: P0 108 / P1 92 / draws 0; no cap hits. (Run 1: 99/101/0.)
- Observation JSON averages 2.9 KiB/step (from measurement 2).

## 2. Single-process step cost, three tiers

Instrumented decomposition, 100 games / 4,755 selections (timed components
cover 95.4 % of wall; ~6 `perf_counter` calls/selection ≈ sub-µs overhead):

| Component | µs/selection | share |
|---|---|---|
| `lib.Select` (C engine: apply action) | 13.5 | 2.8 % |
| `lib.GetBattleData` (C engine: serialize state) | 180.3 | 37.4 % |
| `json.loads` (incl. c_char_p→bytes copy) | 79.9 | 16.6 % |
| `search_begin_input` string copy (game.py always pays) | 6.3 | 1.3 % |
| `to_observation_class` (recursive dataclass) | 201.5 | 41.9 % |

| Tier | µs/selection |
|---|---|
| (a) raw engine (`Select` + `GetBattleData`) | 193.8 |
| (b) + `json.loads` | 273.7 |
| (c) + dataclass conversion | 475.2 |
| full sample-agent path (c + sbi) | 481.5 |

Clean tier (b) throughput window (no per-step instrumentation): **2,053 games,
101,143 selections in 30.02 s → 68.4 games/s, 3,370 selections/s
(296.8 µs/selection incl. Python action sampling)**. Derived tier (c)
equivalent: ≈ 2,000 selections/s ≈ **41 games/s** (0.60× tier b).

## 3. Multi-process scaling (tier b, 30 s windows, spawn)

| Workers | games | games/s | selections/s | efficiency vs 1w | RSS/worker |
|---|---|---|---|---|---|
| 1 | 2,030 | 67.6 | 3,445 | 100 % | 29 MB |
| 2 | 4,524 | 150.7 | 7,493 | 111 % | 29 MB |
| 3 | 5,343 | 178.0 | 8,930 | 88 % | 29 MB |
| 4 | 6,027 | **200.8** | 9,872 | 74 % | 29–30 MB |

Variance re-check (fresh run): 1 worker 74.1 games/s, 2 workers 145.8 games/s
(98 % efficiency). The 111 % above is baseline noise, not real superlinearity —
the 1-worker rate moved 67.6→74.1 (~9 %) between runs (engine-RNG game mix +
scheduling). Reading: **near-linear to 2 workers, knee at 3–4** on this 4-core
no-SMT desktop that is also running the OS and session tooling. Peak aggregate
≈ 200 games/s ≈ **50 games/s per fully-subscribed core** (68–74 games/s on an
isolated core).

## 4. Memory

- RSS after import + `GameInitialize`: 23.9 MiB; after warmup: 24.6 MiB;
  spawn workers under load: **29–30 MB each**.
- Leak check, one process, `battle_finish` every game:
  - 300 games (spec): +1.2 MiB, fitted +3.4 KB/game — but non-monotonic.
  - Extended 3,000 games: 24.6 → 26.7 MiB (+2.2 MiB); overall fit +542
    B/game, **decelerating** — growth is concentrated in the first ~1,500
    games; the back half oscillates at 26.3–26.9 MiB with no upward trend.
- Reading: allocator/arena growth reaching a plateau (~27 MiB), **no unbounded
  leak** at this horizon. Recommend a ≥50k-game soak on the cloud host before
  multi-day runs.

## 5. Stretch: search API throughput

From a turn-9 mid-game snapshot (30 selections in; predictions = count-correct
card lists drawn from the sample deck, basics first), random rollouts for 20 s:

| Metric | Value |
|---|---|
| `search_step` | **2,364/s in-call (423 µs/step**, incl. json→dataclass decode of each returned state) |
| `search_begin` | 0.42 ms/call (n=483) |
| End-to-end rollout throughput | 2,250 steps/s; 93.2 steps per rollout to terminal; 0 cap hits |

## Answers to the four decisions

**1. Games/sec/core and projected cloud throughput.** Tier (b) random
self-play: 68–74 games/s on an isolated core, **~50 games/s/core with all
cores subscribed** (the number that matters for a fleet). Mean game ≈ 15 ms of
engine+parse time. Extrapolation at 50 games/s/core (random-policy,
env-only, ~55-selection games):
- 96 cores: ~4,800 games/s ≈ **4.1×10⁸ games/day**
- 192 cores: ~9,600 games/s ≈ **8.3×10⁸ games/day**

Treat as an upper bound on env throughput: competent policies will play longer
games (throughput scales ~1/selections), server cores may clock lower than
this 3.5–3.9 GHz desktop, and PPO inference — not the env — will almost
certainly be the real bottleneck. Even at 10× derating, the env supports
~4×10⁷ games/day at 96 cores.

**2. Selections/game and the 10-minute clock.** Per player: mean 27.6,
median 21, p95 68, max 101 selections (random play). Only **13 % of selections
are forced** (measured), vs the brief's "~half trivial" assumption. Budget per
selection from 600 s/player, at the worst-side p95 of 68 selections:
**~8.8 s/move flat**; ~10.1 s per non-forced move using the measured 13 %
forced share; ~17.6 s using the brief's half-trivial assumption. Even if
competent play doubles game length, ~4–9 s/move remains — enough for
**~10–20k `search_step` calls per move on one core** (at 2,364 steps/s). RL
episode horizon: plan for ~55 steps mean / ~130 p95 / ~220 max (cap 5,000 was
never approached).

**3. Where time goes → custom parser?** Yes — skip `to_observation_class` in
the RL loop. It is the single largest component (202 µs, 42 % of the full
sample-agent path); dropping it (tier b: engine + `json.loads` dict) gives
**1.6–1.7× more games/s** for free (41 → 68 games/s single-core). The C engine
itself is not the bottleneck you'd expect: applying an action costs 13.5 µs
while serializing the observation to JSON costs 180 µs — so the hard floor
without engine-side changes is tier (a) ≈ 194 µs/selection (~5,200
selections/s/core); JSON decode is unavoidable through this API. A leaner
hand-rolled extractor (pull only `select`, `result`, `yourIndex` from the dict)
can approach tier (b) but not beat tier (a).

**4. Memory per env and parallelism.** ~30 MB RSS per worker process,
plateauing (no unbounded leak over 3,000 games). Memory permits hundreds of
envs per typical node (96 envs ≈ ~3 GB); **CPU, not RAM, is the binding
constraint — size env count to physical cores** (scaling efficiency here: ~100 %
to 2/4 cores, 74 % at 4/4 on a machine also running an OS and desktop session;
expect better residual efficiency on a headless cloud host, minus NUMA
unknowns at 96+ cores).

## Anomalies and honesty notes

- **Engine RNG is not seedable** from the public API → runs are not exactly
  reproducible (documented above; affects games/s at the ±5–9 % level).
- **First timed run of the session was ~50 % slower per-selection** (443 µs vs
  ~293 µs at the same tier) than all subsequent runs; suspected antivirus scan
  of the freshly written files / cold OS caches. It was re-run; the canonical
  structure numbers come from the re-run, which agrees with the decomposition
  and clean-window runs. Both runs are preserved in the log.
- First scaling attempt crashed **after** its timed windows in my RSS helper
  (ctypes `GetProcessMemoryInfo` without explicit prototypes truncates the
  pseudo-handle on x64); fixed with explicit `argtypes`/`restype` and re-run.
  No engine issue.
- No draws in 400 structure games; no selection-cap hits anywhere; no
  `IndexError` from `battle_select` (no illegal-selection bugs surfaced with
  the sample-agent policy).
- Windows Defender real-time protection was active throughout — a modest,
  uncontrolled background factor.

## Re-running

```
python benchmarks/smoke_test.py
python benchmarks/bench_structure.py
python benchmarks/bench_stepcost.py
python benchmarks/bench_scaling.py        # optional args: worker counts
python benchmarks/bench_memory.py         # optional args: n_games sample_every
python benchmarks/bench_search.py
```
