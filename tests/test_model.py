import torch
from ptcg.cards import build_tables
from ptcg.model import Encoder, collate_states, teacher_config, tiny_config


def _real_states(n=4):
    import random
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.featurize import featurize_state
    from ptcg.tracker import BeliefTracker
    tables = build_tables()
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    rng = random.Random(1)
    out = []
    try:
        while len(out) < n and not s.done:
            me = s.select_player
            out.append(featurize_state(s.obs, me, deck,
                                       BeliefTracker(me).snapshot(), tables))
            s.select(random_picks(s.obs, rng))
    finally:
        s.close()
    return tables, out


def test_forward_shapes_and_grads():
    tables, states = _real_states()
    cfg = tiny_config(tables)
    enc = Encoder(cfg)
    batch = collate_states(states)
    h = enc(batch)
    assert h.shape == (len(states), batch["card"].shape[1], cfg.d)
    h.sum().backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


def test_padding_invariance():
    tables, states = _real_states(2)
    enc = Encoder(tiny_config(tables)).eval()
    batch = collate_states(states)
    with torch.no_grad():
        h1 = enc(batch)
        batch2 = {k: v.clone() for k, v in batch.items()}
        batch2["card"][~batch2["mask"]] = 1  # scribble on padding
        h2 = enc(batch2)
    m = batch["mask"]
    assert torch.allclose(h1[m], h2[m], atol=1e-5)


def test_teacher_param_count():
    tables, _ = _real_states(1)
    n = sum(p.numel() for p in Encoder(teacher_config(tables)).parameters())
    assert 20_000_000 < n < 32_000_000


def test_option_logits_masking():
    import numpy as np
    from ptcg.featurize import encode_select, featurize_state
    from ptcg.model import PolicyModel, collate_selects, collate_states, tiny_config
    tables, states = _real_states(1)
    # rebuild the matching select for the same obs — reuse helper game
    import random
    from ptcg.engine import BattleSession, load_sample_deck, random_picks
    from ptcg.tracker import BeliefTracker
    deck = load_sample_deck()
    s = BattleSession(deck, list(deck))
    try:
        me = s.select_player
        ts = featurize_state(s.obs, me, deck, BeliefTracker(me).snapshot(), tables)
        es = encode_select(s.obs, ts, tables)
    finally:
        s.close()
    m = PolicyModel(tiny_config(tables))
    sb = collate_states([ts])
    trunk = m.encode(sb)
    selb = collate_selects([es])
    o = len(es.opt_type)
    picked = torch.zeros((1, o + 1), dtype=torch.bool)
    logits = m.option_logits(trunk, sb, selb, picked)
    assert logits.shape == (1, o + 1)
    if es.min_count >= 1:
        assert logits[0, o] == float("-inf")     # done illegal before min picks
    picked[0, 0] = True
    logits2 = m.option_logits(trunk, sb, selb, picked)
    assert logits2[0, 0] == float("-inf")        # picked option masked


def test_heads_shapes():
    from ptcg.model import PolicyModel, collate_states, tiny_config
    tables, states = _real_states(2)
    m = PolicyModel(tiny_config(tables))
    trunk = m.encode(collate_states(states))
    assert m.public_value(trunk).shape == (2,)
    assert m.public_value(trunk).abs().max() < 1.0
    assert m.aux_decklist(trunk).shape == (2, tables.n_rows)
    assert (m.aux_decklist(trunk) >= 0).all()
