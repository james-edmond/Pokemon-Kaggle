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
