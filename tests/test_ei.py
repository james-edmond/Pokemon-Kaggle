import math
import random

import torch

from ptcg.cards import build_tables
from ptcg.ei import (EIConfig, flatten_games, is_single_pick,
                     pi_targets_single, single_pick_loss, wilson)
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.featurize import encode_select, featurize_state, featurize_privileged
from ptcg.selfplay_search import EIGame, EIStep
from ptcg.tracker import BeliefTracker


def test_wilson_matches_known_value():
    lo, hi = wilson(4, 12)
    assert abs(lo - 0.138) < 2e-3 and abs(hi - 0.609) < 2e-3


def _fabricated_game(tables, n_states=6):
    """Real featurized states from a live battle; fabricated candidates."""
    deck = load_sample_deck()
    for t in range(6):
        s = BattleSession(deck, deck)
        keep = False
        try:
            rng = random.Random(11 + t)
            trk = {0: BeliefTracker(0), 1: BeliefTracker(1)}
            steps = []
            for _ in range(60):
                if s.done or len(steps) >= n_states:
                    break
                me = s.obs["current"]["yourIndex"]
                trk[me].update(s.obs.get("logs") or [])
                sel = s.obs["select"]
                ts = featurize_state(s.obs, me, deck, trk[me].snapshot(),
                                     tables)
                es = encode_select(s.obs, ts, tables)
                pv = featurize_privileged(s.obs, s.obs, (deck, deck), tables)
                n = len(sel["option"])
                if sel["maxCount"] == 1 and n >= 2:
                    acts = [(j,) for j in range(min(n, 3))]
                    vis = [3, 1] + ([1] if len(acts) == 3 else [])
                    steps.append(EIStep(me, ts, es, pv, acts, vis))
                else:
                    steps.append(EIStep(me, ts, es, pv, None, None))
                s.select(random_picks(s.obs, rng))
            if len(steps) >= 3 and any(x.actions for x in steps):
                keep = True
                return EIGame(steps, 0, (1.0, -1.0), (list(deck), list(deck)))
        finally:
            s.close()
    raise AssertionError("no usable fabricated game")


def test_flatten_and_single_pick_partition():
    tables = build_tables()
    g = _fabricated_game(tables)
    flat = flatten_games([g])
    assert len(flat) == len(g.steps)
    st, z, od = flat[0]
    assert z == g.rewards[st.player]
    assert od == g.decks[1 - st.player]
    singles = [s for s, _, _ in flat if s.actions and is_single_pick(s)]
    assert singles


def test_pi_targets_temperature_math():
    tables = build_tables()
    g = _fabricated_game(tables)
    s = next(x for x in g.steps if x.actions and len(x.actions) == 2)
    t1 = pi_targets_single([s], O=8, temp=1.0)
    assert abs(float(t1[0, s.actions[0][0]]) - 0.75) < 1e-6   # visits [3,1]
    assert abs(float(t1[0, s.actions[1][0]]) - 0.25) < 1e-6
    t2 = pi_targets_single([s], O=8, temp=0.5)                # ^(1/T)=^2
    assert abs(float(t2[0, s.actions[0][0]]) - 0.9) < 1e-6    # 9:1
    assert abs(float(t2.sum()) - 1.0) < 1e-6


def test_single_pick_loss_decreases_with_training():
    import os

    from ptcg.model import PolicyModel, tiny_config
    tables = build_tables()
    torch.manual_seed(0)
    net = PolicyModel(tiny_config(tables))
    g = _fabricated_game(tables)
    flat = [x for x in flatten_games([g]) if x[0].actions
            and is_single_pick(x[0])]
    steps = [x[0] for x in flat]
    zs = [x[1] for x in flat]
    ods = [x[2] for x in flat]
    cfg = EIConfig(lr=5e-3, device="cpu")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    first = last = None
    for i in range(30):
        loss, parts = single_pick_loss(net, steps, zs, ods, tables, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, (first, last)
    assert set(parts) >= {"loss_pi", "loss_v", "loss_aux"}
    assert all(math.isfinite(v) for v in parts.values())


def test_multi_pick_loss_matches_manual_logprob_weighting():
    import os

    from ptcg.model import PolicyModel, tiny_config
    from ptcg.action import replay_logprob
    from ptcg.ei import multi_pick_loss
    tables = build_tables()
    torch.manual_seed(1)
    net = PolicyModel(tiny_config(tables))
    g = _fabricated_game(tables)
    s = next(x for x in g.steps if x.actions and len(x.actions) >= 2)
    cfg = EIConfig()
    loss, parts = multi_pick_loss(net, s, 1.0, g.decks[1 - s.player],
                                  tables, cfg)
    # manual: -sum(pi_a * logp_a) via the B==1 replay path
    with torch.no_grad():
        lps = replay_logprob(net, [s.state] * len(s.actions),
                             [s.esel] * len(s.actions),
                             [list(a) for a in s.actions])
    w = torch.tensor([float(v) for v in s.visits])
    pi = w / w.sum()
    manual = -(pi * lps).sum()
    assert abs(float(parts["loss_pi"]) - float(manual)) < 1e-4


def test_train_ei_runs_and_improves_on_fabricated_data():
    from ptcg.model import PolicyModel, tiny_config
    from ptcg.ei import train_ei
    tables = build_tables()
    torch.manual_seed(2)
    net = PolicyModel(tiny_config(tables))
    games = [_fabricated_game(tables) for _ in range(2)]
    cfg = EIConfig(lr=3e-3, epochs=4, minibatch=8, device="cpu", seed=0)
    m1 = train_ei(net, games, tables, cfg)
    m2 = train_ei(net, games, tables, cfg)
    assert m2["loss_pi"] < m1["loss_pi"]
    assert m1["n_single"] + m1["n_multi"] + m1["n_valueonly"] == sum(
        len(g.steps) for g in games)
    for k in ("loss_pi", "loss_v", "loss_aux"):
        assert k in m1
