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
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(0)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    steps = ep.steps[:6]
    deckA = deck
    deckB = list(reversed(deck))                 # a different "opponent" deck id-list
    per = [deckB if s.player == 0 else deckA for s in steps]
    pd, dl, hd = aux_targets(steps, tables, per)
    assert dl.shape == (len(steps), tables.n_rows)
    # a step whose opponent deck is deckB has that deck's row counts
    i = next(k for k, s in enumerate(steps) if s.player == 0)
    from collections import Counter
    exp = Counter(card_row(c, tables.n_rows) for c in deckB)
    assert dl[i, next(iter(exp))] > 0


def test_aux_targets_single_deck_backcompat():
    tables = build_tables()
    deck = load_sample_deck()
    m = PolicyModel(tiny_config(tables))
    g = torch.Generator().manual_seed(1)
    ep = play_league_game(m, m, (deck, list(deck)), tables, learner_seat=0,
                          mirror=True, generator=g)
    pd, dl, hd = aux_targets(ep.steps[:4], tables, deck)   # single list[int]
    assert dl.shape == (4, tables.n_rows)
