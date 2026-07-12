# Phase-5 expert iteration — run playbook

## Stage 1 box
- 64-128 vCPU + 1x 4090/A100-class GPU, Linux x86, >=200 GB disk.
- `git clone` the repo (or rsync the working tree), then
  `bash scripts/cloud_setup.sh cu121` (learner) — generation is CPU-only.
- Prerequisite: copy `champ/phase3-generalist-r120.pt` and
  `submission_src/policy.pt` to the box (both gitignored, so `git clone`
  does NOT bring them) — the start ckpt seeds cycle 0 and the anchor leg of
  the gate needs the phase-3 generalist.

## Launch (detached; never as a child of an agent session)
    mkdir -p runs
    nohup .venv/bin/python scripts/ei_loop.py --run-id ei-a --cycles 5 \
      --start-ckpt submission_src/policy.pt --games-per-cycle 8000 \
      --k 3 --sims 32 --gate-games 300 --gate-raw 60 \
      --workers 60 --device cuda --seed 1 > runs/ei-a.log 2>&1 &

  (8000 games x ~131 moves/game ~= the spec's ~1M-move/cycle target.)

## Resume after any interruption
Re-run the SAME command: completed stages are skipped via their markers
(manifest.json / ckpt.pt / gate.json), the loop continues mid-cycle.

## Artifact sync (run from the LOCAL machine, repeat while the run lives)
    rsync -avz cloud:~/Pokemon-Kaggle/runs/ei/ei-a/ runs/ei/ei-a/ \
      --exclude 'data/worker-*'          # gate/ckpt/state only; add data if wanted

## Costs (measured, d224, k=3/sims=32 generation)
- ~131 moves/game search-vs-search.
- sims-capped generation ~2-4 s/searched move/core (tslice=inf, so the
  k_trees*sims_per_tree budget binds — NOT the old 1-1.5 s figure).
- 64 cores ~= 400-800 games/hr -> an 8k-game cycle ~= 10-20 h ~= $20-60 gen,
  plus a few $ train/gate.
- Storage after the trim+gzip fix: measured 2.94 KB/move on disk (states are
  sliced to their used token rows, then gzipped; gzip crushes the padded/zero
  structure ~16x). An ~1M-move cycle therefore lands ~3 GB on disk, well
  under the old ~215-430 GB projection.
- Trainer RAM is bounded by ONE batch file: train_ei streams files
  (`train_ei_stream`), never loading the whole corpus.
- stage-1 budget envelope ~$100-500; stage-2 trigger per the spec.

## Ship checklist (mid-month + end)
1. promoted ckpt -> `python scripts/extract_policy.py` equivalent: save the
   `"policy"` state dict as submission_src/policy.pt (backup the old one)
2. `python scripts/make_submission.py` and `--no-search` (both must pass)
3. `python scripts/test_submission.py 4 --small-search` (legal, searched>0)
4. `python scripts/gate_ei.py --candidate <new> --incumbent <old policy.pt>
   --out runs/ei/ship-gate.json` — ship only on PROMOTE
5. user uploads dist/submission.zip; rollback = previous zip pair

## Stage triggers (pre-registered, from the spec)
- stage-2 (capacity) iff cycles still PROMOTE but per-cycle search-wrapped
  gains fall below ~+2%: d384/d512 init-by-distillation, distill back to a
  CPU-fast student before shipping.
- abort/diagnose iff two consecutive REJECTs: check entropy, audit.json
  mean_jump trend, replay ratio — one variable at a time.

## Box notes
- `cloud_setup.sh` apt-installs `python3.11`: available on Ubuntu 22.04, but
  24.04 needs the deadsnakes PPA (or just use the system `python3.12` —
  torch 2.5.1 ships 3.12 wheels, so swap `python3.11`->`python3.12` in the
  setup command).
