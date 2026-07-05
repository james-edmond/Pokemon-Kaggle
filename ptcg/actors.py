import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .cards import build_tables
from .engine import BattleSession, load_sample_deck, random_picks
from .featurize import encode_select, featurize_state
from .model import PolicyModel
from .rollout import play_game
from .tracker import BeliefTracker
from .trainloop import (TrainConfig, game_seed, load_checkpoint,
                        model_config_for, round_dir)
from .action import sample_select


class _NullCritic:
    """Checkpoint files carry the critic too; actors don't need it."""

    def load_state_dict(self, sd):
        return None


def collect_round_worker(args):
    cfg_json, round_n, actor_idx, n_games, ckpt_path = args
    cfg = TrainConfig(**json.loads(cfg_json))
    tables = build_tables()
    deck = load_sample_deck()
    policy = PolicyModel(model_config_for(cfg.model_size, tables))
    load_checkpoint(ckpt_path, policy, _NullCritic(), optim=None)
    policy.eval()
    rd = round_dir(cfg, round_n)
    rd.mkdir(parents=True, exist_ok=True)
    from .trainloop import save_game
    t0 = time.perf_counter()
    steps = 0
    results = []
    for g in range(n_games):
        gen = torch.Generator().manual_seed(
            game_seed(cfg, round_n, actor_idx, g))
        # spec's persistent debug sample: raw obs of the first game of every
        # 10th round, captured by actor 0 only
        obs_log = [] if (round_n % 10 == 0 and actor_idx == 0 and g == 0) else None
        with torch.no_grad():
            ep = play_game(policy, (deck, list(deck)), tables, generator=gen,
                           step_cap=cfg.step_cap, priv_viz=True,
                           obs_log=obs_log)
        save_game(rd / f"a{actor_idx}-g{g}.pt", ep)
        if obs_log is not None:
            dbg = Path(cfg.run_dir) / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            torch.save(obs_log, dbg / f"round-{round_n:04d}-g0-obs.pt")
        steps += len(ep.steps)
        results.append(ep.result)
    return {"games": n_games, "steps": steps, "results": results,
            "wall_s": time.perf_counter() - t0}


def run_actor_pool(cfg, round_n, ckpt_path, worker=collect_round_worker,
                   extra=None):
    import multiprocessing as mp
    per = cfg.games_per_round // cfg.actors
    rem = cfg.games_per_round - per * cfg.actors
    jobs = []
    for a in range(cfg.actors):
        n = per + (1 if a < rem else 0)
        if n == 0:
            continue
        base = (json.dumps(asdict(cfg)), round_n, a, n, str(ckpt_path))
        jobs.append(base + tuple(extra or ()))
    if len(jobs) == 1:
        return [worker(jobs[0])]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        return pool.map(worker, jobs)


def play_versus(model, opponent, tables, decks, generator, model_seat):
    s = BattleSession(decks[0], decks[1])
    trackers = (BeliefTracker(0), BeliefTracker(1))
    rng = random.Random(int(torch.randint(1 << 30, (1,), generator=generator)))
    try:
        n = 0
        while not s.done:
            n += 1
            if n > 5000:
                raise RuntimeError("step cap exceeded")
            me = s.select_player
            trackers[me].update(s.obs.get("logs", []))
            actor = model if me == model_seat else opponent
            if actor == "random":
                s.select(random_picks(s.obs, rng))
                continue
            ts = featurize_state(s.obs, me, decks[me],
                                 trackers[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            with torch.no_grad():
                d = sample_select(actor, ts, es, generator)
            s.select(d.picks)
        r = s.result
    finally:
        s.close()
    return 1 if r == model_seat else 0
