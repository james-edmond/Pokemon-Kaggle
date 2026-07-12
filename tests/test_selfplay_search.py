import random

import torch

from ptcg.cards import build_tables
from ptcg.decks import PORTFOLIO
from ptcg.mcts import SearchConfig
from ptcg.selfplay_search import (EIGame, EIStep, play_search_game,
                                  sample_deck_pair)


def _tiny_net(tables):
    import os

    from ptcg.model import PolicyModel, student_config
    m = PolicyModel(student_config(tables))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m.load_state_dict(torch.load(
        os.path.join(repo, "submission_src", "policy.pt"),
        map_location="cpu", weights_only=True))
    m.eval()
    return m


def test_sample_deck_pair_mirror_and_distinct():
    rng = random.Random(0)
    pairs = [sample_deck_pair(rng) for _ in range(200)]
    names = set(PORTFOLIO)
    assert all(a in names and b in names for a, b in pairs)
    mirrors = sum(1 for a, b in pairs if a == b)
    assert 20 <= mirrors <= 120          # ~30% of 200, loose bounds
    assert any(a != b for a, b in pairs)


def test_play_search_game_records_targets_and_outcome():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=2, sims_per_tree=8)
    rng = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=rng, gen=gen)
    assert isinstance(g, EIGame)
    assert g.result in (0, 1, 2)
    assert g.rewards in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
    assert len(g.decks) == 2 and len(g.decks[0]) == 60
    assert len(g.steps) > 0
    searched = [s for s in g.steps if s.actions is not None]
    valueonly = [s for s in g.steps if s.actions is None]
    assert searched, "no move recorded a visit distribution"
    for s in g.steps:
        assert isinstance(s, EIStep)
        assert s.player in (0, 1)
        assert s.priv_state is not None
        if s.actions is not None:
            assert len(s.actions) == len(s.visits) >= 2
            assert all(isinstance(a, tuple) for a in s.actions)
            assert sum(s.visits) > 0
    # aux_targets consumes EIStep unchanged (field-name contract)
    from ptcg.ppo import aux_targets
    opp = [g.decks[1 - s.player] for s in g.steps]
    pd, dl, hd = aux_targets(g.steps, tables, opp)
    assert pd.shape[0] == len(g.steps) and dl.shape[0] == len(g.steps)
    assert valueonly is not None   # trivial/forced moves may or may not occur


def _two_real_states(tables):
    """Two real (public, priv, esel, seat) tuples from a live sample battle."""
    import numpy as np

    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.featurize import (encode_select, featurize_privileged,
                                featurize_state)
    from ptcg.tracker import BeliefTracker
    deck = load_sample_deck()
    out = []
    s = BattleSession(deck, deck)
    trk = {0: BeliefTracker(0), 1: BeliefTracker(1)}
    rng = random.Random(7)
    try:
        while len(out) < 2 and not s.done:
            me = s.obs["current"]["yourIndex"]
            trk[me].update(s.obs.get("logs") or [])
            ts = featurize_state(s.obs, me, deck, trk[me].snapshot(), tables)
            es = encode_select(s.obs, ts, tables)
            pv = featurize_privileged(s.obs, s.obs, (deck, deck), tables)
            out.append((me, ts, es, pv))
            s.select(random_picks(s.obs, rng))
    finally:
        s.close()
    assert len(out) == 2, "sample battle ended before two states"
    return deck, out


def test_trim_state_repad_bit_exact_and_flows_unchanged():
    import numpy as np

    from ptcg.action import sample_select
    from ptcg.model import collate_states
    from ptcg.ppo import aux_targets
    from ptcg.selfplay_search import repad_state, trim_state
    tables = build_tables()
    net = _tiny_net(tables)
    deck, ((m0, a, ea, pa), (m1, b, eb, pb)) = _two_real_states(tables)
    fields = ["card", "numeric", "owner", "zone", "kind", "pos", "mask"]

    # (1) repad(trim(.)) is bit-for-bit identical to the original featurization
    for orig in (a, pa, b, pb):
        r = repad_state(trim_state(orig))
        assert r.n == orig.n
        for f in fields:
            assert np.array_equal(getattr(r, f), getattr(orig, f)), f
        # trimming actually shrank the stored arrays
        assert trim_state(orig).card.shape[0] == orig.n < orig.card.shape[0]

    # (2) collate over repadded-trimmed states == collate over originals
    c_orig = collate_states([a, b])
    c_trim = collate_states([repad_state(trim_state(a)),
                             repad_state(trim_state(b))])
    for k in c_orig:
        assert torch.equal(c_orig[k], c_trim[k]), k

    # (3) aux_targets on trimmed priv/public states == on originals
    orig_steps = [EIStep(m0, a, ea, pa), EIStep(m1, b, eb, pb)]
    trim_steps = [EIStep(m0, trim_state(a), ea, trim_state(pa)),
                  EIStep(m1, trim_state(b), eb, trim_state(pb))]
    opp = [list(deck), list(deck)]
    for x, y in zip(aux_targets(orig_steps, tables, opp),
                    aux_targets(trim_steps, tables, opp)):
        assert torch.equal(x, y)

    # (4) sample_select with a fixed generator returns identical picks
    d1 = sample_select(net, a, ea, torch.Generator().manual_seed(123))
    d2 = sample_select(net, repad_state(trim_state(a)), ea,
                       torch.Generator().manual_seed(123))
    assert d1.picks == d2.picks


def test_play_search_game_record_false_skips_steps():
    tables = build_tables()
    net = _tiny_net(tables)
    cfg = SearchConfig(k_trees=1, sims_per_tree=4)
    g = play_search_game(net, net, ("sample", "sample"), tables,
                         cfg=cfg, rng=random.Random(5),
                         gen=torch.Generator().manual_seed(5), record=False)
    assert g.steps == [] and g.result in (0, 1, 2)
