import torch
from ptcg.cards import build_tables, card_row
from ptcg.ppo import aux_targets
from ptcg.rollout import play_league_game
from ptcg.engine import load_sample_deck
from ptcg.model import PolicyModel, tiny_config


class _S:  # minimal Step stand-in with the fields aux_targets reads
    def __init__(self, player, state, priv):
        self.player, self.state, self.priv_state = player, state, priv


def test_aux_targets_per_step_decks():
    from collections import Counter
    from ptcg.decks import deck as get_deck
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    steps = ep.steps[:6]
    deckA = get_deck("dragapult-ex")            # genuinely different card content
    deckB = get_deck("raging-bolt-ex")
    per = [deckB if s.player == 0 else deckA for s in steps]
    pd, dl, hd = aux_targets(steps, tables, per)
    assert dl.shape == (len(steps), tables.n_rows)
    cntA = Counter(card_row(c, tables.n_rows) for c in deckA)
    cntB = Counter(card_row(c, tables.n_rows) for c in deckB)
    disc = [r for r in set(cntA) | set(cntB) if cntA.get(r, 0) != cntB.get(r, 0)]
    assert disc, "decks not distinguishable at card-row granularity"
    row = disc[0]
    # each step's decklist target must reflect ITS assigned opponent deck
    for i, s in enumerate(steps):
        want = cntB if s.player == 0 else cntA
        assert dl[i, row].item() == float(want.get(row, 0)), (i, s.player)


def test_aux_targets_single_deck_backcompat():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    pd, dl, hd = aux_targets(ep.steps[:4], tables, deck)   # single list[int]
    assert dl.shape == (4, tables.n_rows)
