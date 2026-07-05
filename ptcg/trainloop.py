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


def learner_update(policy, critic, optim, episodes, cfg, tables, opp_deck):
    import torch as _t
    from .model import collate_selects, collate_states
    from .ppo import (assemble_advantages, aux_targets, ppo_policy_loss)
    from .replay import batched_replay

    device = _t.device(cfg.device)
    policy.to(device)
    critic.to(device)
    steps, old_lp, adv, ret = assemble_advantages(
        episodes, critic, device=device, lam=cfg.lam, gamma=cfg.gamma)
    pd_t, dl_t, hd_t = aux_targets(steps, tables, opp_deck)
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

    # epoch-0 ratio gate over the full round
    with _t.no_grad():
        drifts = []
        for idx in minibatches(list(range(B))):
            _, _, lp, _ = replay_mb(idx, grad=False)
            drifts.append((lp - old_lp[idx]).exp().sub(1).abs().max())
        ratio_drift = float(_t.stack(drifts).max()) if drifts else 0.0
    if ratio_drift > cfg.ratio_gate:
        raise RuntimeError(
            f"ratio gate violated: max|ratio-1|={ratio_drift:.2e} "
            f"> {cfg.ratio_gate:.0e} — policy/data mismatch")

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
