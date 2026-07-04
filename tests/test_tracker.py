from ptcg.tracker import (AREA_DECK, AREA_HAND, LOG_ATTACH, LOG_DRAW_REVERSE,
                          LOG_EVOLVE, LOG_MOVE_CARD, LOG_MOVE_CARD_REVERSE,
                          LOG_PLAY, BeliefTracker)

AREA_DISCARD = 3


def mv(p, cid, fr, to):
    return {"type": LOG_MOVE_CARD, "playerIndex": p, "cardId": cid,
            "fromArea": fr, "toArea": to}


def mvr(p, fr, to):
    return {"type": LOG_MOVE_CARD_REVERSE, "playerIndex": p,
            "fromArea": fr, "toArea": to}


def test_tutor_reveal_then_play():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 112, AREA_DECK, AREA_HAND)])       # opp tutors Munkidori to hand
    assert t.snapshot().opp_hand == {112: 1}
    t.update([mv(1, 112, AREA_HAND, AREA_DISCARD)])    # opp plays/discards it
    assert t.snapshot().opp_hand == {}


def test_own_moves_ignored():
    t = BeliefTracker(my_index=0)
    t.update([mv(0, 112, AREA_DECK, AREA_HAND)])
    assert t.snapshot().opp_hand == {}


def test_facedown_hand_to_deck_demotes_to_pool():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 140, AREA_DECK, AREA_HAND)])
    t.update([mvr(1, AREA_HAND, AREA_DECK)])           # Iono-style facedown return
    s = t.snapshot()
    assert s.opp_hand == {} and s.opp_hidden_pool == {140: 1}


def test_unknown_log_type_ignored():
    t = BeliefTracker(my_index=0)
    t.update([{"type": 9999, "playerIndex": 1}, {"type": LOG_MOVE_CARD}])
    assert t.snapshot().opp_hand == {}


def test_opp_play_log_removes_phantom_hand_entry():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 112, AREA_DECK, AREA_HAND)])       # tutor reveal to hand
    assert t.snapshot().opp_hand == {112: 1}
    t.update([{"type": LOG_PLAY, "playerIndex": 1, "cardId": 112, "serial": 9}])
    assert t.snapshot().opp_hand == {}                 # no phantom left behind


def test_opp_attach_and_evolve_logs_decrement_hand():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 3, AREA_DECK, AREA_HAND), mv(1, 745, AREA_DECK, AREA_HAND)])
    t.update([{"type": LOG_ATTACH, "playerIndex": 1, "cardId": 3,
               "cardIdTarget": 700}])
    t.update([{"type": LOG_EVOLVE, "playerIndex": 1, "cardId": 745,
               "cardIdTarget": 700}])
    assert t.snapshot().opp_hand == {}


def test_play_log_falls_back_to_hidden_pool():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 140, AREA_DECK, AREA_HAND)])
    t.update([mvr(1, AREA_HAND, AREA_DECK)])           # demoted to pool
    assert t.snapshot().opp_hidden_pool == {140: 1}
    t.update([{"type": LOG_PLAY, "playerIndex": 1, "cardId": 140}])
    assert t.snapshot().opp_hidden_pool == {}


def test_draw_reverse_demotes_deck_knowledge_to_pool():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 140, AREA_HAND, AREA_DECK)])       # known card back to deck
    assert t.snapshot().opp_deck == {140: 1}
    t.update([{"type": LOG_DRAW_REVERSE, "playerIndex": 1}])
    s = t.snapshot()
    assert s.opp_deck == {} and s.opp_hidden_pool == {140: 1}


def test_own_play_and_draw_reverse_ignored():
    t = BeliefTracker(my_index=0)
    t.update([mv(1, 112, AREA_DECK, AREA_HAND),
              mv(1, 140, AREA_HAND, AREA_DECK)])
    t.update([{"type": LOG_PLAY, "playerIndex": 0, "cardId": 112},
              {"type": LOG_DRAW_REVERSE, "playerIndex": 0}])
    s = t.snapshot()
    assert s.opp_hand == {112: 1} and s.opp_deck == {140: 1}
