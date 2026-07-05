# Phase-2 qualification run

Launch (training venv, repo root):

    venv-train\Scripts\python scripts\train.py --run-id phase2-a --device cuda --seed 1 --minibatch 128

(`--device cpu` if benchmarks/RESULTS-gpu.md recorded the CPU fallback.)
`--minibatch 128` is required on this 6 GiB card: learner memory scales
with minibatch, and the TrainConfig default of 512 exceeds 6 GiB VRAM
(the 2026-07-05 launch OOM'd in the round-0 learner update at 12.46 GiB
allocated; measured peaks on that round's real data: 7.96 GiB at 256,
6.06 GiB at 192, 4.14 GiB at 128 — only 128 stays inside a 4.5 GiB
true-VRAM budget and avoids the slow sysmem spill).
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

The train loop prunes automatically after every round
(`prune_checkpoints` in `ptcg/trainloop.py`, called right after the
metrics row is appended): it deletes every checkpoint that is not round 0,
not a multiple of `eval_every` (the `wr_ck5`/`wr_ck15` eval references
read checkpoints 5 and 15 rounds back, always multiples of 5 given
`eval_every=5`), and not one of the two most recent. No manual cleanup is
needed. Round-in-progress data under `runs/<run-id>/rounds/` is transient
and deleted automatically after each round — never needs manual cleanup.

Disk math with pruning: kept checkpoints after N rounds ≈ N/5 + 3, so at
~103 MB each a 72 h run (~430-720 rounds at 6-10 min/round) retains
~90-150 checkpoints ≈ 9-15 GB, and even 1000 rounds stay ≈ 21 GB — well
inside the ~43.5 GB free measured at launch time (2026-07-05). Without
pruning the same run would need 44-74 GB and would not fit. Still monitor
free space periodically; the multiples-of-5 tier keeps growing for the
life of the run.
