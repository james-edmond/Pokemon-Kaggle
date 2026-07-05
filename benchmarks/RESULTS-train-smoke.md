# Training pipeline smoke runs

Date: 2026-07-05

Machine: i5-6600K (4C/4T), 16 GiB RAM, GTX 1060 6GB, Windows 10 Pro 10.0.19045.

## CPU smoke (base interpreter, real spawn multiprocessing)

Interpreter: `C:\Python314\python.exe` (Python 3.14.6, torch 2.12.1+cpu).

```
python scripts/train.py --run-id smoke-cpu --model-size tiny --games-per-round 6 --actors 2 --max-rounds 2 --minibatch 64 --epochs 1 --eval-every 2 --eval-games-random 4 --eval-games-ckpt 2 --device cpu
```

Total wall time: **1 m 08 s** (exit 0). This was the first real exercise of
the spawn `mp.Pool` path (2 actor processes per round, each with its own
engine); no pickling or `__main__`-guard issues on Windows.

| round | kind | games | steps | mean_len | wall_s | ratio_drift | approx_kl |
|---|---|---|---|---|---|---|---|
| 0 | train | 6 | 356 | 59.3 | 29 | 4.76837e-07 | 0.00409 |
| 1 | train | 6 | 418 | 69.7 | 27 | 4.76837e-07 | 0.01011 |
| 1 | eval | 4 (random) | | | | | wr_random 0.750 ± 0.424 |

Artifacts as expected: `checkpoint-0000..0002.pt`, `metrics.csv` with 2 train
rows + 1 eval row, `rounds/` left empty (each round's episode files consumed
and deleted), `debug/round-0000-g0-obs.pt` (the round-0 obs sample from
actor 0). No `wr_ck5`/`wr_ck15` columns — reference checkpoints 5/15 rounds
back don't exist yet, which is the designed behavior.

Plots: `python scripts/plot_run.py runs/smoke-cpu` wrote `plots/train.png`
(all four panels populated) and `plots/eval.png` (single wr_random point with
the 0.5/0.65 guide lines). matplotlib 3.11.0 was installed into the base env
for this (`python -m pip install matplotlib`); the venv already had it.

## GPU smoke (training venv, CPU-actor / GPU-learner path)

Interpreter: `venv-train\Scripts\python` (Python 3.12.10, torch 2.5.1+cu121,
device cuda per benchmarks/RESULTS-gpu.md).

```
venv-train\Scripts\python scripts\train.py --run-id smoke-gpu --model-size student --games-per-round 12 --actors 3 --max-rounds 1 --minibatch 256 --eval-every 999 --device cuda
```

Total wall time: **37.5 s** (exit 0). 3 spawn actor processes collected on
CPU; the learner replayed and updated on cuda.

| round | games | steps | mean_len | wall_s | ratio_drift | epochs_ran |
|---|---|---|---|---|---|---|
| 0 | 12 | 591 | 49.2 | 32 | **7.15256e-07** | 1 |

**ratio_drift = 7.15256e-07** — the first real cross-config measurement of
CPU-rollout vs GPU-replay exactness, more than three orders of magnitude
below the 1e-3 abort gate. The float32 CPU/CUDA kernel differences are at the
~1e-6 relative level, as hoped; the gate has ample headroom.

All losses finite: loss_pg 0.0294, loss_v 0.346, loss_critic 0.332,
loss_aux 3.93, entropy 1.475, approx_kl 0.0358. epochs_ran is 1 of a possible
2 because mean epoch KL (0.0358) exceeded kl_stop (0.02) — the early-stop
working as designed on a cold-start policy, not a failure.

The only console output was the known-benign `enable_nested_tensor`
UserWarning (already documented in RESULTS-gpu.md and suppressed in pytest
config).

## Actor throughput (extrapolated)

Derived from round wall time, which includes the learner update and
checkpoint save, so these are conservative lower bounds on pure actor
throughput. Cold-start games are short (mean_len 49–70 selections);
throughput will drop as the policy learns to play longer games.

| config | games | round wall | games/hour |
|---|---|---|---|
| tiny, 2 CPU actors (base 3.14) | 12 over 2 rounds | 29 s + 27 s | ~770 |
| student, 3 CPU actors + cuda learner | 12 | 32 s | ~1350 |

At ~1350 games/hour the phase-2 default of 192 games/round implies roughly
8–9 minutes of collection per round at cold-start game lengths.
