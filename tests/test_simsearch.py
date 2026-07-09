import random
from types import SimpleNamespace

from ptcg.engine import BattleSession, load_sample_deck, random_picks
from ptcg.simsearch import SearchSession


def _truth_det(sess, obs):
    """Determinization from visualize_data ground truth (test-only)."""
    viz = sess.viz_current()
    me = obs["current"]["yourIndex"]
    vme, vopp = viz["players"][me], viz["players"][1 - me]
    ids = lambda z: [c["id"] for c in z]
    opp_active = []
    oa = obs["current"]["players"][1 - me].get("active") or []
    if oa and oa[0] is None:
        opp_active = [ids(vopp["active"])[0]]
    return SimpleNamespace(
        your_deck=ids(vme["deck"]), your_prize=ids(vme["prize"]),
        opp_deck=ids(vopp["deck"]), opp_prize=ids(vopp["prize"]),
        opp_hand=ids(vopp["hand"]), opp_active=opp_active)


def test_search_session_round_trip_and_errors():
    deck = load_sample_deck()
    sess = BattleSession(deck, deck)
    rng = random.Random(7)
    try:
        for _ in range(30):
            if sess.done:
                break
            sess.select(random_picks(sess.obs, rng))
        assert not sess.done
        obs = sess.obs
        det = _truth_det(sess, obs)
        ss = SearchSession()
        assert ss.ensure_ptr() is True

        # begin: root select mirrors the live select
        got = ss.begin(obs, det)
        assert got is not None
        sid, robs = got
        assert len(robs["select"]["option"]) == len(obs["select"]["option"])
        assert robs["current"]["yourIndex"] == obs["current"]["yourIndex"]

        # two roots coexist; stepping the first still works
        got2 = ss.begin(obs, det)
        assert got2 is not None and got2[0] != sid
        sel = robs["select"]
        child = ss.step(sid, [0] if sel["minCount"] >= 1 else [])
        assert child is not None and child[0] not in (sid, got2[0])

        # illegal picks -> None, not an exception
        assert ss.step(sid, [len(sel["option"]) + 5]) is None

        # walk to terminal; stepping past it -> None
        node_id, nobs = child
        r = random.Random(3)
        for _ in range(400):
            if nobs["current"]["result"] != -1:
                break
            s = nobs["select"]
            k = r.randint(s["minCount"], s["maxCount"])
            nxt = ss.step(node_id, r.sample(range(len(s["option"])), k))
            assert nxt is not None
            node_id, nobs = nxt
        assert nobs["current"]["result"] != -1
        assert ss.step(node_id, [0]) is None

        # too-short arrays are rejected BEFORE the C call
        bad = SimpleNamespace(your_deck=[], your_prize=det.your_prize,
                              opp_deck=det.opp_deck, opp_prize=det.opp_prize,
                              opp_hand=det.opp_hand, opp_active=det.opp_active)
        assert ss.begin(obs, bad) is None

        # obs without search_begin_input -> None
        stripped = dict(obs)
        stripped.pop("search_begin_input", None)
        assert ss.begin(stripped, det) is None

        ss.end()
        assert ss.step(sid, [0]) is None   # released arena -> None, no raise
    finally:
        sess.close()
