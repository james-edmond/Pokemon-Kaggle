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


class _FakeSession3Ply:
    """3-ply game for the phase-consistent leaf rule.
    root sid 0 (me, seat 0) -(1,)-> sid 2 (opponent, seat 1, non-terminal)
    -(1,)-> sid 4 (me, seat 0, non-terminal)."""

    def __init__(self):
        self.tree = {
            (0, (1,)): (2, _obs(seat=1, result=-1)),
            (2, (1,)): (4, _obs(seat=0, result=-1)),
        }

    def step(self, sid, picks):
        return self.tree.get((sid, tuple(picks)))

    def end(self):
        pass


class _FakeSessionOppTerminal:
    """root (me) -(1,)-> opponent node -(0,)-> TERMINAL, opponent wins."""

    def __init__(self):
        self.tree = {
            (0, (1,)): (2, _obs(seat=1, result=-1)),
            (2, (0,)): (3, _obs(seat=1, result=1)),
        }

    def step(self, sid, picks):
        return self.tree.get((sid, tuple(picks)))

    def end(self):
        pass


class _FakeSessionInfiniteCorridor:
    """Every step returns a fresh non-terminal opponent-seat obs: an engine
    turn that (hypothetically) never ends, to exercise the depth bound."""

    def step(self, sid, picks):
        return (0, _obs(seat=1, result=-1))

    def end(self):
        pass


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
    # Phase-consistent rule: a leaf value is only trustworthy read from MY
    # seat (the value head has a turn-phase level shift between turn-start
    # and mid-turn reads). An opponent node is expanded for actions/priors
    # only -- its v (0.77) must be discarded, not backed up in either sign
    # -- and the descent continues (via PUCT on the freshly expanded, all
    # -zero-N node) down to a my-seat leaf, whose v (0.9) is the one and
    # only value that gets backed up, my-phase-consistent at every level.
    def fake_eval(model, obs, seat, deck, belief, tables, gen, m):
        if obs["current"]["yourIndex"] == 1:        # opponent node
            return [(0,), (1,)], [0.0, 1.0], 0.77    # v must be ignored
        return [(0,)], [1.0], 0.9                    # my node: real leaf v
    monkeypatch.setattr(M, "_eval_state", fake_eval)
    sess = _FakeSession3Ply()
    root = _mk_root(sess)
    root.P = [0.0, 1.0]                              # force branch (1,)
    ran = M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                      M.SearchConfig())
    assert ran
    assert root.N == [0, 1]
    # my-phase leaf value (+0.9), NOT -0.77 (opponent's raw v, sign-flipped)
    # and NOT -0.9 (a joint leaf+backup double sign-flip of the my-leaf v)
    assert abs(root.W[1] - 0.9) < 1e-9

    opp_node = root.children[1]
    assert opp_node.actions == [(0,), (1,)]          # expanded, not a leaf
    assert opp_node.N == [0, 1]
    assert abs(opp_node.W[1] - (-0.9)) < 1e-9         # negamax from its seat

    my_leaf = opp_node.children[1]
    assert my_leaf.actions == [(0,)]                 # populated from my eval


def test_simulate_opp_node_never_evaluated_as_leaf(monkeypatch):
    # The opp node's own eval value (0.9) must never reach any W: only the
    # exact terminal outcome one ply further may back up a value.
    monkeypatch.setattr(M, "_eval_state",
                        lambda *a: ([(0,)], [1.0], 0.9))
    sess = _FakeSessionOppTerminal()
    root = _mk_root(sess)
    root.P = [0.0, 1.0]                              # force branch (1,)
    M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                M.SearchConfig())
    assert root.N == [0, 1]
    assert abs(root.W[1] - (-1.0)) < 1e-9             # exact terminal, not 0.9
    opp_node = root.children[1]
    assert opp_node.N == [1]
    assert abs(opp_node.W[0] - 1.0) < 1e-9             # negamax of the -1.0


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


def test_simulate_depth_bound_neutral(monkeypatch):
    # An engine turn is finite, but the descent must not hang if it isn't:
    # a corridor of ever-fresh opponent-seat nodes never reaches a my-seat
    # leaf or a terminal, so the depth bound must cut it off and back up a
    # neutral 0.0 instead of looping forever.
    monkeypatch.setattr(M, "_eval_state",
                        lambda *a: ([(0,)], [1.0], 0.9))
    sess = _FakeSessionInfiniteCorridor()
    root = _mk_root(sess)
    root.P = [0.0, 1.0]                              # force branch (1,)
    ran = M._simulate(root, 0, [3] * 60, [3] * 60, None, None, sess, None,
                      M.SearchConfig())
    assert ran
    assert root.N[1] == 1
    assert root.W[1] == 0.0


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
            assert stats.root_actions is not None
            assert stats.root_visits is not None
            assert len(stats.root_actions) == len(stats.root_visits)
            assert sum(stats.root_visits) == stats.sims
            assert tuple(picks) in stats.root_actions
        else:
            # single-action selects shortcut without searching
            assert stats.reason == "single-action" and picks is not None
        # slice + one leaf-eval of overshoot is the budget contract
        assert dt < 4.0 + 2.5, f"took {dt:.1f}s"
        assert stats.elapsed <= dt
    finally:
        sess.close()


def test_vote_counts_exposes_summed_visits():
    sess = _FakeSession()
    a = _mk_root(sess)
    b = _mk_root(sess)
    a.N, a.W = [3, 1], [1.0, 0.5]
    b.N, b.W = [1, 3], [0.2, 0.4]
    votes, wsum = M._vote_counts([(a, None), (b, None)])
    assert votes[(0,)] == 4 and votes[(1,)] == 4
    assert abs(wsum[(0,)] - 1.2) < 1e-9 and abs(wsum[(1,)] - 0.9) < 1e-9
    # _vote must agree with the counts it is built on
    assert M._vote([(a, None), (b, None)]) in ((0,), (1,))


def test_movestats_has_root_fields_defaulting_none():
    st = M.MoveStats()
    assert st.root_actions is None and st.root_visits is None
