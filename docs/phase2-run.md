# Phase-2 qualification run

Launch (training venv, repo root):

    venv-train\Scripts\python scripts\train.py --run-id phase2-a --device cuda --seed 1

(`--device cpu` if benchmarks/RESULTS-gpu.md recorded the CPU fallback.)
Resume after any interruption: run the same command — the loop continues
from the last complete checkpoint.

Round cadence: ~10-60 min per round depending on game length as play
improves (measured ~9 min/round at cold-start, student size, cuda, 192
games/round — see benchmarks/RESULTS-train-smoke.md — not the flat ~1 h a
naive estimate would suggest).

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

## Disk

Student checkpoints are ~103 MB each (`checkpoint-*.pt`, includes optimizer
state); a multi-day qualification run accumulates tens of GB under
`runs/<run-id>/` if every round's checkpoint is kept. `runs/` is
git-ignored, so none of this is tracked by git.

Safe to delete: any checkpoint that is not round 0, not a multiple of 5
(the `wr_ck5`/`wr_ck15` eval references read checkpoints 5 and 15 rounds
back, which are always multiples of 5 given `eval_every=5`), and not one of
the two most recent rounds. Round-in-progress data under
`runs/<run-id>/rounds/` is transient and deleted automatically after each
round — never needs manual cleanup.

At launch time (2026-07-05) this machine had ~43.5 GB free on `C:` —
below the 60 GB comfort margin for a multi-day run. Monitor free space
periodically and prune old checkpoints (per the rule above) if it drops
further.
