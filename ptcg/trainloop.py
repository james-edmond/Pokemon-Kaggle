import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .featurize import FEATURIZER_VERSION
from .model import (CriticModel, PolicyModel, critic_config, student_config,
                    teacher_config, tiny_config)


@dataclass
class TrainConfig:
    run_dir: str
    model_size: str = "student"
    games_per_round: int = 192
    actors: int = 3
    epochs: int = 2
    minibatch: int = 512
    lr: float = 3e-4
    clip: float = 0.2
    lam: float = 0.95
    gamma: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    critic_coef: float = 0.5
    aux_coef: float = 0.1
    kl_stop: float = 0.02
    grad_clip: float = 1.0
    ratio_gate: float = 1e-3
    eval_every: int = 5
    eval_games_random: int = 200
    eval_games_ckpt: int = 100
    device: str = "cpu"
    seed: int = 0
    step_cap: int = 5000
    mirror_frac: float = 0.30
    pool_frac: float = 0.65
    random_frac: float = 0.05
    snapshot_every: int = 5
    pool_cap: int = 18
    sd_champ_ckpt: str = ""


def model_config_for(size, tables):
    return {"tiny": tiny_config, "student": student_config,
            "teacher": teacher_config}[size](tables)


def round_dir(cfg, n) -> Path:
    return Path(cfg.run_dir) / "rounds" / str(n)


def checkpoint_path(cfg, n) -> Path:
    return Path(cfg.run_dir) / f"checkpoint-{n:04d}.pt"


def game_seed(cfg, round_n, actor_idx, game_idx) -> int:
    return ((cfg.seed * 1_000_003 + round_n) * 101 + actor_idx) * 10_007 + game_idx


def save_game(path, episode) -> None:
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(episode, tmp)
    os.replace(tmp, path)


def load_round(cfg, n):
    rd = round_dir(cfg, n)
    eps = []
    for f in sorted(rd.glob("*.pt")):
        eps.append(torch.load(f, weights_only=False))
    return eps


def save_checkpoint(cfg, n, policy, critic, optim) -> None:
    path = checkpoint_path(cfg, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "round": n,
        "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
        "optim": optim.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "config": asdict(cfg),
        "featurizer_version": FEATURIZER_VERSION,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, critic, optim=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["policy"])
    critic.load_state_dict(ck["critic"])
    if optim is not None:
        optim.load_state_dict(ck["optim"])
        torch.set_rng_state(ck["torch_rng"])
    return ck["round"]


def latest_checkpoint(cfg):
    best = None
    for f in Path(cfg.run_dir).glob("checkpoint-*.pt"):
        n = int(f.stem.split("-")[1])
        if best is None or n > best[0]:
            best = (n, f)
    return best


def prune_checkpoints(cfg, current_round) -> list[str]:
    """Delete checkpoints that are not round 0, not multiples of cfg.eval_every
    (eval reference candidates), and not one of the two most recent. Returns
    deleted filenames."""
    ckpts = []
    for f in Path(cfg.run_dir).glob("checkpoint-*.pt"):
        try:
            n = int(f.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if n <= current_round + 1:  # never touch anything newer than now
            ckpts.append((n, f))
    recent = set(sorted(n for n, _ in ckpts)[-2:])  # two most recent
    deleted = []
    for n, f in sorted(ckpts):
        if n == 0 or n % cfg.eval_every == 0 or n in recent:
            continue
        f.unlink()
        deleted.append(f.name)
    return sorted(deleted)


def learner_update(policy, critic, optim, episodes, cfg, tables, opp_deck):
    import torch as _t
    from .model import collate_selects, collate_states
    from .ppo import (assemble_advantages, aux_targets, ppo_policy_loss,
                      ratio_drift_stats)
    from .replay import batched_replay

    device = _t.device(cfg.device)
    policy.to(device)
    critic.to(device)
    steps, old_lp, adv, ret = assemble_advantages(
        episodes, critic, device=device, lam=cfg.lam, gamma=cfg.gamma)
    # build per-step opponent decks from each episode's recorded decks
    opp_decks = []
    for ep in episodes:
        ep_decks = getattr(ep, "decks", (None, None))
        for s in ep.steps:
            od = ep_decks[1 - s.player] if ep_decks[1 - s.player] is not None else opp_deck
            opp_decks.append(od)
    assert len(opp_decks) == len(steps)
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_decks)
    old_lp, adv, ret = old_lp.to(device), adv.to(device), ret.to(device)
    pd_t, dl_t, hd_t = pd_t.to(device), dl_t.to(device), hd_t.to(device)
    B = len(steps)

    def minibatches(order):
        for lo in range(0, B, cfg.minibatch):
            yield order[lo:lo + cfg.minibatch]

    def replay_mb(idx, grad):
        sb = collate_states([steps[i].state for i in idx])
        selb = collate_selects([steps[i].esel for i in idx])
        sb = {k: v.to(device) for k, v in sb.items()}
        selb = {k: v.to(device) for k, v in selb.items()}
        ctx = _t.enable_grad() if grad else _t.no_grad()
        with ctx:
            trunk = policy.encode(sb)
            lp, ent = batched_replay(policy, trunk, sb, selb,
                                     [steps[i].picks for i in idx])
        return sb, trunk, lp, ent

    # epoch-0 ratio gate over the full round. Abort on the MEAN |ratio-1|: a
    # genuine policy/data mismatch shifts every step, whereas benign
    # CPU-collect/GPU-replay divergence spikes only the single worst step (and
    # grows with policy sharpness over a long run). ratio_drift (logged) stays
    # the max for diagnostic visibility.
    with _t.no_grad():
        dmax, dsum, dn = 0.0, 0.0, 0
        for idx in minibatches(list(range(B))):
            _, _, lp, _ = replay_mb(idx, grad=False)
            mx, sm, n = ratio_drift_stats(lp, old_lp[idx])
            dmax = max(dmax, mx)
            dsum += sm
            dn += n
        ratio_drift = dmax
        ratio_mean = dsum / max(dn, 1)
    if ratio_mean > cfg.ratio_gate:
        raise RuntimeError(
            f"ratio gate violated: mean|ratio-1|={ratio_mean:.2e} "
            f"> {cfg.ratio_gate:.0e} (max={ratio_drift:.2e}) — policy/data mismatch")

    poiss = _t.nn.PoissonNLLLoss(log_input=False, full=False)
    agg = {k: 0.0 for k in ("loss_pg", "loss_v", "loss_critic", "loss_aux",
                            "entropy", "approx_kl")}
    n_mb = 0
    epochs_ran = 0
    gen = _t.Generator().manual_seed(cfg.seed + 17)
    for epoch in range(cfg.epochs):
        order = _t.randperm(B, generator=gen).tolist()
        kl_epoch = []
        for idx in minibatches(order):
            sb, trunk, lp, ent = replay_mb(idx, grad=True)
            pg, ratio, kl = ppo_policy_loss(
                lp, old_lp[idx], adv[idx], clip=cfg.clip)
            v = policy.public_value(trunk)
            vloss = ((v - ret[idx]) ** 2).mean()
            pvb = collate_states([steps[i].priv_state for i in idx])
            pvb = {k: t.to(device) for k, t in pvb.items()}
            cvals = critic(pvb)
            players = _t.tensor([steps[i].player for i in idx], device=device)
            closs = ((cvals[_t.arange(len(idx), device=device), players]
                      - ret[idx]) ** 2).mean()
            aux = (((policy.prize_diff(trunk) - pd_t[idx]) ** 2).mean()
                   + poiss(policy.aux_decklist(trunk), dl_t[idx])
                   + poiss(policy.aux_hand(trunk), hd_t[idx]))
            loss = (pg + cfg.vf_coef * vloss + cfg.critic_coef * closs
                    - cfg.ent_coef * ent.mean() + cfg.aux_coef * aux)
            optim.zero_grad()
            loss.backward()
            _t.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()),
                cfg.grad_clip)
            optim.step()
            for k, val in (("loss_pg", pg), ("loss_v", vloss),
                           ("loss_critic", closs), ("loss_aux", aux),
                           ("entropy", ent.mean()), ("approx_kl", kl)):
                agg[k] += float(val.detach())
            kl_epoch.append(float(kl.detach()))
            n_mb += 1
        epochs_ran += 1
        if kl_epoch and sum(kl_epoch) / len(kl_epoch) > cfg.kl_stop:
            break
    out = {k: v / max(n_mb, 1) for k, v in agg.items()}
    out.update(epochs_ran=epochs_ran, ratio_drift=ratio_drift, steps=B)
    return out


import csv
import shutil

METRIC_FIELDS = ["round", "kind", "games", "steps", "loss_pg", "loss_v",
                 "loss_critic", "loss_aux", "entropy", "approx_kl",
                 "epochs_ran", "ratio_drift", "wr_random", "ci_random",
                 "wr_ck5", "wr_ck15", "wr_random_mean", "wr_champ_nonsample",
                 "wr_champ_sample", "mean_len", "wall_s"]


def _metrics_path(cfg):
    return Path(cfg.run_dir) / "metrics.csv"


def append_metrics(cfg, row):
    p = _metrics_path(cfg)
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def read_metrics(cfg):
    p = _metrics_path(cfg)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def truncate_metrics(cfg, before_round):
    rows = [r for r in read_metrics(cfg) if int(r["round"]) < before_round]
    p = _metrics_path(cfg)
    tmp = p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)


def _config_drift_warnings(ck_config, cfg) -> list[str]:
    """Compare a checkpoint's stored config against the live cfg and return
    human-readable warnings (no raise) for drift on keys that change training
    semantics, plus a warning when eval_every departs from the assumed 5."""
    warnings = []
    cur = asdict(cfg)
    for key in ("model_size", "minibatch", "eval_every",
                "games_per_round", "device"):
        a = ck_config.get(key)
        b = cur.get(key)
        if a != b:
            warnings.append(
                f"warning: resume config drift: {key} checkpoint={a} cli={b}")
    if cfg.eval_every != 5:
        warnings.append(
            f"warning: eval_every={cfg.eval_every} != 5; eval reference "
            "offsets (5/15) and pruning protection assume eval_every=5")
    return warnings


def _eval_due(cfg, round_n, policy, tables):
    if (round_n + 1) % cfg.eval_every != 0:
        return None
    import math

    from .actors import eval_worker, league_eval_worker, run_actor_pool
    from .decks import SAMPLE, all_decks
    ck = checkpoint_path(cfg, round_n + 1)
    row = {"round": round_n, "kind": "eval"}

    def _run(n_games, opp_spec):
        sub = TrainConfig(**{**asdict(cfg), "games_per_round": n_games})
        stats = run_actor_pool(sub, round_n, ck, worker=eval_worker,
                               extra=(opp_spec,))
        wins = sum(s["wins"] for s in stats)
        games = sum(s["games"] for s in stats)
        return wins / max(games, 1), games

    def _run_deck(n_games, opp_spec, deck_name):
        sub = TrainConfig(**{**asdict(cfg), "games_per_round": n_games})
        stats = run_actor_pool(sub, round_n, ck, worker=league_eval_worker,
                               extra=(opp_spec, deck_name))
        wins = sum(s["wins"] for s in stats)
        games = sum(s["games"] for s in stats)
        return wins / max(games, 1)

    # existing sample-deck metrics (UNCHANGED)
    wr, n = _run(cfg.eval_games_random, "random")
    row["wr_random"] = f"{wr:.3f}"
    row["ci_random"] = f"{1.96 * math.sqrt(max(wr * (1 - wr), 1e-9) / max(n, 1)):.3f}"
    for label, back in (("wr_ck5", 5), ("wr_ck15", 15)):
        ref = checkpoint_path(cfg, round_n + 1 - back)
        if ref.exists():
            wr, _ = _run(cfg.eval_games_ckpt, str(ref))
            row[label] = f"{wr:.3f}"

    # per-deck vs random over the portfolio -> wr_random_mean
    names = all_decks()
    per_random = max(1, cfg.eval_games_random // len(names))
    rand_rates = [_run_deck(per_random, "random", nm) for nm in names]
    row["wr_random_mean"] = f"{sum(rand_rates) / len(rand_rates):.3f}"

    # vs frozen SD-champ over the portfolio (only if the champ exists)
    champ = cfg.sd_champ_ckpt
    if champ and Path(champ).exists():
        per_champ = max(1, cfg.eval_games_ckpt // len(names))
        nonsample = []
        for nm in names:
            r = _run_deck(per_champ, champ, nm)
            if nm == SAMPLE:
                row["wr_champ_sample"] = f"{r:.3f}"
            else:
                nonsample.append(r)
        if nonsample:
            row["wr_champ_nonsample"] = f"{sum(nonsample) / len(nonsample):.3f}"
    return row


def train(cfg, max_rounds):
    import time as _time

    from .actors import league_round_worker, run_actor_pool
    from .cards import build_tables
    from .engine import load_sample_deck
    from .league import prune_pool, snapshot

    tables = build_tables()
    deck = load_sample_deck()
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    policy = PolicyModel(model_config_for(cfg.model_size, tables)).to(cfg.device)
    critic = CriticModel(critic_config(tables)).to(cfg.device)
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=cfg.lr)

    # Params live on cfg.device before any checkpoint load so that
    # optim.load_state_dict casts the restored Adam state (exp_avg/exp_avg_sq)
    # onto the params' device automatically; step counters stay on CPU as
    # torch manages non-capturable state_steps. This makes the .to(device)
    # calls inside learner_update idempotent no-ops on resume.
    latest = latest_checkpoint(cfg)
    if latest is None:
        torch.manual_seed(cfg.seed)
        with open(Path(cfg.run_dir) / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=1)
        save_checkpoint(cfg, 0, policy, critic, optim)
        start = 0
    else:
        ck_meta = torch.load(latest[1], map_location="cpu", weights_only=False)
        for w in _config_drift_warnings(ck_meta.get("config", {}), cfg):
            print(w)
        start = load_checkpoint(latest[1], policy, critic, optim)
        truncate_metrics(cfg, start)
        rd = round_dir(cfg, start)
        if rd.exists():
            shutil.rmtree(rd)  # incomplete round from a crash
        # Backfill an eval lost to a kill in the post-checkpoint eval window:
        # the eval for checkpoint `start` runs after that checkpoint is saved,
        # so a kill there loses only the metrics row (the checkpoint is intact).
        if (start > 0 and start % cfg.eval_every == 0
                and checkpoint_path(cfg, start).exists()):
            has_eval = any(r["kind"] == "eval" and int(r["round"]) == start - 1
                           for r in read_metrics(cfg))
            if not has_eval:
                ev = _eval_due(cfg, start - 1, policy, tables)
                if ev is not None:
                    append_metrics(cfg, ev)

    for rnd in range(start, max_rounds):
        t0 = _time.perf_counter()
        ck = checkpoint_path(cfg, rnd)
        stats = run_actor_pool(cfg, rnd, ck, worker=league_round_worker)
        episodes = load_round(cfg, rnd)
        policy.train()
        critic.train()
        m = learner_update(policy, critic, optim, episodes, cfg, tables, deck)
        policy.eval()
        critic.eval()
        save_checkpoint(cfg, rnd + 1, policy, critic, optim)
        if (rnd + 1) % cfg.snapshot_every == 0:
            snapshot(cfg, rnd + 1, policy)
            prune_pool(cfg, cfg.pool_cap)
        n_steps = sum(s["steps"] for s in stats)
        mean_len = n_steps / max(sum(s["games"] for s in stats), 1)
        append_metrics(cfg, dict(
            round=rnd, kind="train", games=sum(s["games"] for s in stats),
            mean_len=f"{mean_len:.1f}", wall_s=f"{_time.perf_counter() - t0:.0f}",
            **{k: (f"{v:.6g}" if isinstance(v, float) else v)
               for k, v in m.items()}))
        prune_checkpoints(cfg, rnd)
        shutil.rmtree(round_dir(cfg, rnd), ignore_errors=True)
        ev = _eval_due(cfg, rnd, policy, tables)
        if ev is not None:
            append_metrics(cfg, ev)
