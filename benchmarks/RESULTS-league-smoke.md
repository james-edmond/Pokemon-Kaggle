# Phase-3 league training: CLI verification, SD-champ freeze, smoke runs

Date: 2026-07-06

Machine: i5-6600K (4C/4T), 16 GiB RAM, GTX 1060 6GB, Windows 10 Pro 10.0.19045.

## SD-champ freeze

```
python scripts/freeze_champ.py runs/phase2-a/checkpoint-0031.pt champ/sd-champ.pt
```

Output: `froze runs/phase2-a/checkpoint-0031.pt -> champ/sd-champ.pt`. Result:
`champ/sd-champ.pt`, 103,484,230 bytes (~103.5 MB), matching the expected
phase-2 checkpoint size. `champ/` added to `.gitignore` (same treatment as
`runs/`); the directory and file are not committed.

## CLI verification

`python scripts/train.py --help` confirms `scripts/train.py` needed no code
change — flags auto-extend from `TrainConfig` fields and already include the
phase-3 additions:

```
  --mirror-frac MIRROR_FRAC
  --pool-frac POOL_FRAC
  --random-frac RANDOM_FRAC
  --snapshot-every SNAPSHOT_EVERY
  --pool-cap POOL_CAP
  --sd-champ-ckpt SD_CHAMP_CKPT
```

## `scripts/plot_run.py` change

Inside the existing `if eval_rows:` block, after the `eval.png` savefig, added
a third figure (`generalization.png`) plotting `wr_champ_nonsample` (headline),
`wr_champ_sample`, and `wr_random_mean` vs round, with a 0.60 dotted
reference line (generalization criterion) and a 0.5 dashed gray baseline.
Reuses the existing `series()` helper (skips rows with empty values), so it
degrades gracefully on runs without a champ configured (CPU smoke below).
`train.png` / `eval.png` logic is unchanged.

## CPU smoke (base interpreter, real spawn multiprocessing)

Interpreter: base `python` (Python 3.14.6, torch 2.12.1+cpu).

```
python scripts/train.py --run-id league-smoke --model-size tiny --games-per-round 8 --actors 2 --max-rounds 3 --minibatch 64 --epochs 1 --snapshot-every 1 --eval-every 2 --eval-games-random 8 --eval-games-ckpt 4 --mirror-frac 0.34 --pool-frac 0.66 --random-frac 0.0 --device cpu
```

Total wall time: **2 m 48 s** (exit 0). No `--sd-champ-ckpt` passed, so the
champ-eval branch is intentionally skipped here (exercised in the GPU smoke
instead).

| round | kind | games | steps | wall_s | ratio_drift | wr_random | ci_random | wr_random_mean |
|---|---|---|---|---|---|---|---|---|
| 0 | train | 8 | 1360 | 72 | 7.15256e-07 | | | |
| 1 | train | 8 | 900 | 52 | 4.76837e-07 | | | |
| 1 | eval | | | | | 0.500 | 0.346 | **0.778** |
| 2 | train | 8 | 545 | 32 | 4.76837e-07 | | | |

Artifacts as expected: `runs/league-smoke/checkpoint-0000.pt` /
`checkpoint-0002.pt` / `checkpoint-0003.pt`, `runs/league-smoke/league/`
holds `snap-0001.pt`, `snap-0002.pt`, `snap-0003.pt` (one per round —
`snapshot-every 1` over 3 rounds), `metrics.csv` has 3 train rows + 1 eval
row with `wr_random_mean` populated (`wr_champ_nonsample`/`wr_champ_sample`
correctly blank since no champ was configured), and
`runs/league-smoke/rounds/` exists but is **empty** — each round's episode
files are consumed and deleted, matching the phase-2 smoke's documented
behavior (no residue).

`python scripts/plot_run.py runs/league-smoke` → `wrote runs\league-smoke\plots`,
producing `train.png`, `eval.png`, and the new `generalization.png` (empty of
champ curves as expected, since none were configured for this run).

## GPU smoke (training venv) — exercises the SD-champ eval path

Interpreter: `venv-train\Scripts\python` (Python 3.12.10, torch 2.5.1+cu121,
device cuda). `torch.cuda.is_available()` verified `True` on the GTX 1060
before the run (qualification run confirmed stopped, GPU free).

**Deviation from the task brief:** the brief's Step 4 command used
`--eval-every 999` (no eval at all). Per this task's refined instructions, an
eval was enabled instead (`--eval-every 2 --eval-games-random 8
--eval-games-ckpt 4 --sd-champ-ckpt champ\sd-champ.pt`) so the
vs-SD-champ path (`wr_champ_nonsample`/`wr_champ_sample`) — the headline
phase-3 generalization metric — is actually exercised on the real `student`
model size before the long qualification run, rather than deferring that
risk to Task 10.

```
venv-train\Scripts\python scripts\train.py --run-id league-gpu --model-size student --games-per-round 12 --actors 3 --max-rounds 2 --minibatch 128 --snapshot-every 1 --eval-every 2 --eval-games-random 8 --eval-games-ckpt 4 --sd-champ-ckpt champ\sd-champ.pt --device cuda
```

Total wall time: **2 m 14.5 s** (exit 0). 3 spawn actor processes collected on
CPU; the learner replayed and updated on cuda; the eval fired after round 1
and successfully loaded `champ/sd-champ.pt` into `league_eval_worker` (no
shape-mismatch or load errors).

| round | kind | games | steps | wall_s | epochs_ran | ratio_drift |
|---|---|---|---|---|---|---|
| 0 | train | 12 | 1525 | 45 | 1 | **9.53674e-07** |
| 1 | train | 12 | 1267 | 51 | 2 | **1.96695e-06** |

Both well under the 1e-3 `ratio_drift` abort gate. All losses finite on both
rows (e.g. round 1: loss_pg -0.00285, loss_v 0.158, loss_critic 0.0937,
loss_aux 1.244, entropy 1.377, approx_kl 0.0117).

Round-1 eval row:

| wr_random | ci_random | wr_random_mean | **wr_champ_nonsample** | **wr_champ_sample** |
|---|---|---|---|---|
| 0.625 | 0.335 | 0.556 | **0.625** | **0.000** |

The champ path ran end to end and populated both headline fields. Note on
magnitude: the portfolio (`ptcg.decks.all_decks()`) has 9 decks, and
`per_champ = max(1, eval_games_ckpt // len(names))` = `max(1, 4 // 9)` = 1
game per deck vs. the frozen champ at this tiny smoke scale. So
`wr_champ_sample` (the `sample` deck alone) is a single win/loss (0/1 here),
and `wr_champ_nonsample` averages 8 single-game results (5/8 = 0.625) — both
numbers are expected to be noisy at this game count and are not a signal
about real generalization strength; they only confirm the plumbing (config
threading, checkpoint loading, per-deck aggregation) is correct. The real
qualification run (Task 10) uses far larger `eval-games-ckpt` for a
statistically meaningful read.

Artifacts as expected: `runs/league-gpu/checkpoint-0000.pt` /
`checkpoint-0001.pt` / `checkpoint-0002.pt`, `runs/league-gpu/league/` holds
`snap-0001.pt`, `snap-0002.pt` (one per round), and
`runs/league-gpu/rounds/` exists but is empty (no residue), matching the CPU
smoke. `python scripts/plot_run.py runs/league-gpu` rendered `train.png`,
`eval.png`, and `generalization.png` (this time with the champ curves
populated).

## Cleanup

Both smoke run directories (`runs/league-smoke/`, `runs/league-gpu/`) were
deleted after recording the results above; they were never intended to be
kept (`runs/` is git-ignored regardless). `champ/sd-champ.pt` was left in
place — the phase-3 qualification run (Task 10) depends on it.
