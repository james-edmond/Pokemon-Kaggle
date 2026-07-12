"""Resumable expert-iteration driver: generate -> train -> gate -> promote.
Each stage is a subprocess (one engine process at a time) and is SKIPPED
when its completion marker exists, so re-running the same command resumes
from the interrupted stage. A REJECTed cycle keeps the incumbent and the
loop continues with fresh data next cycle.
Usage:
  python scripts/ei_loop.py --run-id ei-a --cycles 3 \
      --start-ckpt submission_src/policy.pt --games-per-cycle 200 \
      --k 3 --sims 32 --gate-games 300 --gate-raw 60 --workers 3
State: runs/ei/<run-id>/state.json ; artifacts under cycle-<n>/."""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


@dataclass
class LoopArgs:
    run_dir: str
    start_ckpt: str
    games_per_cycle: int
    k: int
    sims: int
    gate_games: int
    gate_raw: int
    workers: int
    kl_first_cycles: int = 2
    kl_coef: float = 0.02
    device: str = "cpu"
    seed: int = 1


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _run(runner, cmd):
    r = runner(cmd)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"stage failed: {' '.join(str(c) for c in cmd)}")


def run_cycle(run_dir, n, incumbent, a, runner=None):
    """Returns (new_incumbent, promoted_bool). Idempotent per stage."""
    runner = runner or (lambda cmd: subprocess.run(cmd))
    cdir = os.path.join(run_dir, f"cycle-{n}")
    data = os.path.join(cdir, "data")
    ckpt = os.path.join(cdir, "ckpt.pt")
    gate = os.path.join(cdir, "gate.json")
    audit = os.path.join(cdir, "audit.json")
    py = sys.executable
    if not os.path.exists(os.path.join(data, "manifest.json")):
        _run(runner, [py, os.path.join(SCRIPTS, "gen_ei.py"),
                      "--ckpt", incumbent, "--out", data,
                      "--games", str(a.games_per_cycle),
                      "--workers", str(a.workers), "--k", str(a.k),
                      "--sims", str(a.sims), "--seed", str(a.seed + n)])
    if not os.path.exists(ckpt):
        cmd = [py, os.path.join(SCRIPTS, "train_ei.py"),
               "--data", data, "--ckpt-in", incumbent, "--ckpt-out", ckpt,
               "--device", a.device, "--seed", str(a.seed + n)]
        prev = os.path.join(run_dir, f"cycle-{n - 1}", "data")
        if n > 0 and os.path.isdir(prev):
            cmd += ["--replay", prev]
        if n < a.kl_first_cycles:
            cmd += ["--kl-coef", str(a.kl_coef)]
        _run(runner, cmd)
    if not os.path.exists(gate):
        _run(runner, [py, os.path.join(SCRIPTS, "gate_ei.py"),
                      "--candidate", ckpt, "--incumbent", incumbent,
                      "--out", gate, "--games-search", str(a.gate_games),
                      "--games-raw", str(a.gate_raw), "--k", str(a.k),
                      "--sims", str(a.sims), "--workers", str(a.workers),
                      "--seed", str(a.seed + 10 * n)])
    if not os.path.exists(audit):
        _run(runner, [py, os.path.join(SCRIPTS, "audit_value_phase.py"),
                      "--ckpt", ckpt, "--out", audit])
    with open(gate) as f:
        promoted = bool(json.load(f).get("promote"))
    return (ckpt if promoted else incumbent), promoted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--start-ckpt", required=True)
    ap.add_argument("--games-per-cycle", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--gate-games", type=int, default=300)
    ap.add_argument("--gate-raw", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--kl-first-cycles", type=int, default=2)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ag = ap.parse_args()
    run_dir = os.path.join(REPO, "runs", "ei", ag.run_id)
    sp = os.path.join(run_dir, "state.json")
    st = load_state(sp) or {"cycle": 0, "incumbent": ag.start_ckpt,
                            "history": []}
    a = LoopArgs(run_dir, ag.start_ckpt, ag.games_per_cycle, ag.k, ag.sims,
                 ag.gate_games, ag.gate_raw, ag.workers, ag.kl_first_cycles,
                 ag.kl_coef, ag.device, ag.seed)
    while st["cycle"] < ag.cycles:
        n = st["cycle"]
        print(f"ei_loop: cycle {n} incumbent={st['incumbent']}", flush=True)
        inc, promoted = run_cycle(run_dir, n, st["incumbent"], a)
        st["history"].append({"cycle": n, "promoted": promoted,
                              "incumbent_after": inc})
        st["incumbent"] = inc
        st["cycle"] = n + 1
        save_state(sp, st)
    print("ei_loop: done", json.dumps(st))


if __name__ == "__main__":
    main()
