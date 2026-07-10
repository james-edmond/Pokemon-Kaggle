import math

import ptcg.mcts as M


class _FakeSession:
    """Scripted 2-ply game. sid 0 = root (my seat 0).
    action (0,) -> sid 1: terminal, I win.
    action (1,) -> sid 2: opponent node (seat 1) with 2 actions."""

    def __init__(self):
        self.tree = {
            (0, (0,)): (1, _obs(seat=0, result=0)),           # I win
            (0, (1,)): (2, _obs(seat=1, result=-1)),          # opp to move
            (2, (0,)): (3, _obs(seat=0, result=1)),           # opp wins
            (2, (1,)): (4, _obs(seat=0, result=-1)),          # play on
        }

    def step(self, sid, picks):
        return self.tree.get((sid, tuple(picks)))

    def end(self):
        pass


def _obs(seat, result):
    return {"current": {"yourIndex": seat, "result": result,
                        "players": [{}, {}]},
            "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
            "logs": []}


class _FakeTracker:
    def __init__(self, me):
        self.me = me
        self._hand = {}
        self._deck = {}
        self._pool = {}

    def update(self, logs):
        pass

    def snapshot(self):
        return None


def _mk_root(session):
    root = M._Node(0, _obs(seat=0, result=-1), 0,
                   (_FakeTracker(0), _FakeTracker(1)))
    root.actions = [(0,), (1,)]
    root.P = [0.5, 0.5]
    root.N = [0, 0]
    root.W = [0.0, 0.0]
    return root


def test_select_action_puct_math():
    root = _mk_root(_FakeSession())
    root.N = [3, 1]
    root.W = [1.5, 0.9]
    c = 1.5
    tot = math.sqrt(4 + 1)
    s0 = 0.5 + c * 0.5 * tot / 4
    s1 = 0.9 + c * 0.5 * tot / 2
    want = 0 if s0 >= s1 else 1
    assert M._select_action(root, c) == want


def test_simulate_backs_up_negamax(monkeypatch):
    # opponent leaf evaluates +0.9 FOR THE OPPONENT -> -0.9 for me at root
    def fake_eval(model, obs, seat, deck, belief, tables, gen, m):
        return [(0,), (1,)], [0.5, 0.5], 0.9
    monkeypatch.setattr(M, "_eval_state", fake_eval)
    sess = _FakeSession()
    root = _mk_root(sess)
    # force the (1,) branch: bias priors
    root.P = [0.0, 1.0]
    ran = M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                      M.SearchConfig())
    assert ran
    assert root.N == [0, 1]
    assert abs(root.W[1] - (-0.9)) < 1e-9      # flipped into my perspective


def test_simulate_terminal_and_dead_edge(monkeypatch):
    monkeypatch.setattr(M, "_eval_state",
                        lambda *a: ([(0,)], [1.0], 0.0))
    sess = _FakeSession()
    root = _mk_root(sess)
    root.P = [1.0, 0.0]                         # go to the winning terminal
    M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                M.SearchConfig())
    assert root.N[0] == 1 and abs(root.W[0] - 1.0) < 1e-9
    # engine refusing a step becomes a neutral dead edge, not a crash
    root2 = _mk_root(sess)
    root2.actions = [(9,), (0,)]                # (9,) unknown to the fake
    root2.P = [1.0, 0.0]
    root2.N = [0, 0]
    root2.W = [0.0, 0.0]
    M._simulate(root2, 0, [3] * 60, [3] * 60, None, None, sess, None,
                M.SearchConfig())
    assert root2.N[0] == 1 and root2.W[0] == 0.0


def test_vote_sums_across_trees_and_breaks_ties_by_value():
    sess = _FakeSession()
    a = _mk_root(sess)
    b = _mk_root(sess)
    a.N, a.W = [3, 1], [1.0, 0.5]
    b.N, b.W = [1, 3], [0.2, 0.4]
    assert M._vote([(a, None), (b, None)]) in ((0,), (1,))
    b.N = [1, 5]                                # (1,) now dominates 4 vs 6
    assert M._vote([(a, None), (b, None)]) == (1,)
    # tie on visits -> higher mean value wins
    c1, c2 = _mk_root(sess), _mk_root(sess)
    c1.N, c1.W = [2, 2], [1.8, 0.2]
    assert M._vote([(c1, None)]) == (0,)


import os
import random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _search_ready_session(max_tries=6):
    """A live battle warmed up to a not-done state parked on our own select.

    Returns (sess, tracker, me, deck); raises AssertionError if no fresh
    battle reaches a usable state within max_tries attempts.

    Mirrors test_simsearch.py's _mid_game_session / test_determinize.py's
    _probe_session: the engine's internal RNG is unseedable, so a fixed
    python seed does not give a fixed trajectory, and the random warm-up
    below occasionally runs the game to completion before parking on our
    select. Retry with a fresh battle rather than weakening any assertion
    on the probed state itself.
    """
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.tracker import BeliefTracker

    deck = load_sample_deck()
    for t in range(max_tries):
        sess = BattleSession(deck, deck)
        keep = False
        try:
            rng = random.Random(21 + t)
            me = sess.obs["current"]["yourIndex"]
            tracker = BeliefTracker(me)
            for _ in range(20):
                if sess.done:
                    break
                if sess.obs["current"]["yourIndex"] == me:
                    tracker.update(sess.obs.get("logs") or [])
                sess.select(random_picks(sess.obs, rng))
            while sess.obs["current"]["yourIndex"] != me and not sess.done:
                sess.select(random_picks(sess.obs, rng))
            if sess.done:
                continue
            tracker.update(sess.obs.get("logs") or [])
            keep = True
            return sess, tracker, me, deck
        finally:
            if not keep:
                sess.close()
    raise AssertionError("no usable pre-search state in %d tries" % max_tries)


def test_search_move_live_engine_legal_and_budgeted():
    import torch

    from ptcg.cards import build_tables
    from ptcg.model import PolicyModel, student_config
    from ptcg.simsearch import SearchSession

    tables = build_tables()
    model = PolicyModel(student_config(tables))
    model.load_state_dict(torch.load(
        os.path.join(_REPO, "submission_src", "policy.pt"),
        map_location="cpu", weights_only=True))
    model.eval()

    sess, tracker, me, deck = _search_ready_session()
    try:
        obs = sess.obs

        ss = SearchSession()
        cfg = M.SearchConfig(k_trees=2, sims_per_tree=8)
        gen = torch.Generator().manual_seed(0)
        t0 = __import__("time").perf_counter()
        picks, stats = M.search_move(obs, me, deck, tracker, model, tables,
                                     ss, cfg, random.Random(1), gen,
                                     tslice=4.0)
        dt = __import__("time").perf_counter() - t0
        sel = obs["select"]
        if stats.searched:
            assert isinstance(picks, list)
            assert sel["minCount"] <= len(picks) <= sel["maxCount"]
            assert len(set(picks)) == len(picks)
            assert all(0 <= p < len(sel["option"]) for p in picks)
            assert stats.sims >= 1 and stats.trees >= 1
        else:
            # single-action selects shortcut without searching
            assert stats.reason == "single-action" and picks is not None
        # slice + one leaf-eval of overshoot is the budget contract
        assert dt < 4.0 + 2.5, f"took {dt:.1f}s"
        assert stats.elapsed <= dt
    finally:
        sess.close()
