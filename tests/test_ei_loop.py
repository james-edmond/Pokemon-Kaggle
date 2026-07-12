import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import ei_loop


def _fake_runner_factory(promote, calls):
    def fake_run(cmd, **kw):
        calls.append(cmd)
        # emulate each stage's completion marker
        joined = " ".join(str(c) for c in cmd)
        if "gen_ei.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump({"games": 1, "moves": 1}, f)
        elif "train_ei.py" in joined:
            out = cmd[cmd.index("--ckpt-out") + 1]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write("ckpt")
        elif "gate_ei.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            with open(out, "w") as f:
                json.dump({"promote": promote}, f)
        elif "audit_value_phase.py" in joined:
            out = cmd[cmd.index("--out") + 1]
            with open(out, "w") as f:
                json.dump({"mean_jump": 0.1}, f)

        class R:
            returncode = 0
        return R()
    return fake_run


def _args(tmp):
    return ei_loop.LoopArgs(
        run_dir=str(tmp), start_ckpt="start.pt", games_per_cycle=1,
        k=1, sims=4, gate_games=4, gate_raw=2, workers=1,
        kl_first_cycles=1, kl_coef=0.02, device="cpu", seed=1)


def test_promote_advances_incumbent_and_resume_skips(tmp_path):
    calls = []
    a = _args(tmp_path)
    inc, verdict = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                     runner=_fake_runner_factory(True, calls))
    assert verdict is True
    assert inc == os.path.join(str(tmp_path), "cycle-0", "ckpt.pt")
    n_first = len(calls)
    assert n_first >= 3
    # resume: markers exist -> zero subprocess calls, same outcome
    calls2 = []
    inc2, verdict2 = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                       runner=_fake_runner_factory(True, calls2))
    assert (inc2, verdict2) == (inc, True)
    assert calls2 == []


def test_reject_keeps_incumbent(tmp_path):
    calls = []
    a = _args(tmp_path)
    inc, verdict = ei_loop.run_cycle(str(tmp_path), 0, "start.pt", a,
                                     runner=_fake_runner_factory(False, calls))
    assert verdict is False
    assert inc == "start.pt"


def test_state_roundtrip(tmp_path):
    p = os.path.join(str(tmp_path), "state.json")
    st = {"cycle": 2, "incumbent": "x.pt", "history": [{"cycle": 0}]}
    ei_loop.save_state(p, st)
    assert ei_loop.load_state(p) == st
    assert ei_loop.load_state(os.path.join(str(tmp_path), "nope.json")) is None
