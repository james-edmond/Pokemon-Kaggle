from collections import Counter
from ptcg.decks import PORTFOLIO, SAMPLE, all_decks, train_decks, deck, is_legal


def test_portfolio_has_diverse_validated_decks():
    assert SAMPLE in PORTFOLIO
    assert len(PORTFOLIO) >= 6, PORTFOLIO.keys()
    for name in all_decks():
        d = deck(name)
        assert len(d) == 60, (name, len(d))
        ok, why = is_legal(d)
        assert ok, (name, why)          # every committed deck is engine-legal
    # decks are actually distinct (not the same list under many names)
    sigs = {name: tuple(sorted(Counter(deck(name)).items())) for name in all_decks()}
    assert len(set(sigs.values())) == len(sigs), "portfolio decks not distinct"


def test_train_decks_cover_portfolio():
    assert set(train_decks()) == set(all_decks())   # phase 3: all decks train
