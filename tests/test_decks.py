from ptcg.decks import (card_name_index, resolve, deck_from_counts,
                        is_legal, is_playable, validate)
from ptcg.engine import load_sample_deck


def test_name_index_and_resolve():
    idx = card_name_index()
    assert len(idx) > 500                    # real card pool
    # sample deck is Mega Abomasnow ex line; those names must resolve
    assert resolve("Mega Abomasnow ex") is not None
    assert resolve("  mega abomasnow EX  ") == resolve("Mega Abomasnow ex")  # normalized
    assert resolve("Not A Real Card 9999") is None


def test_sample_deck_is_legal_and_playable():
    deck = load_sample_deck()
    ok, why = is_legal(deck)
    assert ok, why
    ok, why = is_playable(deck, n_games=4, seed=1)
    assert ok, why
    assert validate(deck)[0]


def test_illegal_deck_rejected():
    # 60 copies of one non-energy card is not a legal deck
    bad = [resolve("Mega Abomasnow ex")] * 60
    ok, why = is_legal(bad)
    assert not ok and why  # engine rejects; reason non-empty


def test_deck_from_counts_builds_60():
    deck = load_sample_deck()
    # round-trip a known-good structure: rebuild the sample deck from its counts
    from collections import Counter
    # deck_from_counts needs names; use ids directly via a numeric passthrough
    # (deck_from_counts accepts already-int "names" too — see impl)
    counts = [(str(c), n) for c, n in Counter(deck).items()]
    rebuilt = deck_from_counts(counts)
    assert len(rebuilt) == 60 and Counter(rebuilt) == Counter(deck)
