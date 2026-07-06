# Phase-3 qualification run

Prereqs: freeze the SD-champ once —
`venv-train\Scripts\python scripts\freeze_champ.py runs\phase2-a\checkpoint-0031.pt champ\sd-champ.pt`
(if `runs/phase2-a` was deleted, resume phase-2 briefly to regenerate a
checkpoint, or point at any saved phase-2 checkpoint).

Launch FULLY DETACHED (never as a child of a transient shell/agent — a phase-2
lesson), from the repo root:

    venv-train\Scripts\python scripts\train.py --run-id phase3-a --device cuda --seed 1 --minibatch 128 --sd-champ-ckpt champ\sd-champ.pt --games-per-round 256

Resume after any interruption: the same command (loads the latest checkpoint and
the on-disk league pool).

Monitor: `python scripts/plot_run.py runs/phase3-a`

Success criteria (spec §Success criteria — all four, sized-sample stats):
1. Generalization: wr_champ_nonsample > 0.60 on two consecutive evals.
2. Sustained gradient: current beats the 15-rounds-back league snapshot > 0.55
   for materially longer than the single-deck mirror did (which stalled ~round
   10-15).
3. Per-deck competence: >= 0.70 vs random on every trained deck by the end.
4. Stability: zero ratio-gate aborts, zero NaN, a clean kill+resume with
   contiguous metrics AND an intact league pool.

Disk: student checkpoints ~103 MB (pruned to {0, mult of 5, last 2}); league
snapshots are policy-only (~52 MB) capped at pool_cap (~18) with anchor
retention — budget ~10-15 GB. `runs/` and `champ/` are git-ignored.

On success: record run id, final eval row, and wall-clock in
benchmarks/RESULTS-league-smoke.md under a "qualification" heading. On failure:
capture metrics.csv + plots and open a debugging session — do not reinterpret
the criteria.
