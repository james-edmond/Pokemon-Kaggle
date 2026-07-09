import random
from collections import Counter

import numpy as np

from ptcg.cards import build_tables
from ptcg.determinize import (Determinization, filler_determinization,
                              sample_determinization)
from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.simsearch import SearchSession
from ptcg.tracker import BeliefTracker


def _advance(sess, tracker, me, rng, n):
    for _ in range(n):
        if sess.done:
            break
        if sess.obs["current"]["yourIndex"] == me:
            tracker.update(sess.obs.get("logs") or [])
        sess.select(random_picks(sess.obs, rng))


def _counts_ok(det, obs, me):
    cur = obs["current"]
    you, opp = cur["players"][me], cur["players"][1 - me]
    assert len(det.your_deck) == you["deckCount"]
    assert len(det.your_prize) == len(you["prize"] or [])
    assert len(det.opp_deck) == opp["deckCount"]
    assert len(det.opp_prize) == len(opp["prize"] or [])
    assert len(det.opp_hand) == opp["handCount"]
    assert len(det.opp_decklist) == 60
    for c in det.your_deck + det.your_prize + det.opp_deck + det.opp_prize + det.opp_hand:
        assert isinstance(c, int) and c >= 1


def _probe_session(max_tries=6):
    """A live battle advanced to a state where it is OUR turn to select, for
    the determinization probe. Returns (sess, tracker, me, deck).

    Mirrors test_simsearch.py's _mid_game_session: the engine's internal RNG
    is unseedable, so a fixed python seed does not give a fixed trajectory.
    The 25-ply random-pick warm-up, or the post-warm-up park-on-our-select
    loop, occasionally run the game to completion before reaching a usable
    probe point; retry with a fresh battle rather than weakening any
    assertion on the probed state itself.
    """
    deck = load_sample_deck()
    for t in range(max_tries):
        sess = BattleSession(deck, deck)
        keep = False
        try:
            rng = random.Random(11 + t)
            me = sess.obs["current"]["yourIndex"]
            tracker = BeliefTracker(me)
            _advance(sess, tracker, me, rng, 25)
            if sess.done:
                continue
            # park on one of OUR selects
            while sess.obs["current"]["yourIndex"] != me:
                sess.select(random_picks(sess.obs, rng))
                if sess.done:
                    break
            if sess.done:
                continue
            tracker.update(sess.obs.get("logs") or [])
            keep = True
            return sess, tracker, me, deck
        finally:
            if not keep:
                sess.close()
    raise AssertionError("no usable probe state in %d tries" % max_tries)


def test_determinizations_consistent_and_engine_accepted():
    tables = build_tables()
    sess, tracker, me, deck = _probe_session()
    try:
        obs = sess.obs
        belief = tracker.snapshot()

        n_rows = tables.n_rows
        r = np.random.RandomState(0)
        dl = np.abs(r.normal(0.05, 0.05, n_rows)).astype(np.float32)
        for cid in deck:                       # informed-ish decklist prior
            dl[cid + 2] += 1.0
        hd = np.abs(r.normal(0.02, 0.02, n_rows)).astype(np.float32)

        ss = SearchSession()
        srng = random.Random(5)
        accepted = 0
        for i in range(50):
            det = sample_determinization(obs, me, deck, belief, dl, hd,
                                         tables, srng)
            _counts_ok(det, obs, me)
            # tracker known-hand cards are a hard minimum of the sampled hand
            hand_c = Counter(det.opp_hand)
            for cid, n in belief.opp_hand.items():
                assert hand_c[cid] >= n, (cid, n, hand_c)
            # revealed opp prizes preserved at their indices
            for j, c in enumerate(obs["current"]["players"][1 - me]["prize"] or []):
                if c is not None:
                    assert det.opp_prize[j] == c["id"]
            if ss.begin(obs, det) is not None:
                accepted += 1
        ss.end()
        assert accepted == 50, f"engine rejected {50 - accepted}/50 samples"

        # filler fallback: exact counts, engine-accepted, no aux/belief needed
        det = filler_determinization(obs, me, deck, tables, srng)
        _counts_ok(det, obs, me)
        got = ss.begin(obs, det)
        assert got is not None
        ss.end()
    finally:
        sess.close()


def test_never_raises_on_garbage_obs():
    tables = build_tables()
    rng = random.Random(0)
    for obs in ({}, {"current": None}, {"current": {"players": []}},
                {"current": {"players": [{}, {}], "yourIndex": 0},
                 "select": {}}):
        d = sample_determinization(obs, 0, [3] * 60, None, None, None,
                                   tables, rng)
        assert isinstance(d, Determinization)
        d2 = filler_determinization(obs, 0, [3] * 60, tables, rng)
        assert isinstance(d2, Determinization)
