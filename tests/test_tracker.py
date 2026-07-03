from ptcg.tracker import BeliefTracker, LOG_MOVE_CARD, LOG_MOVE_CARD_REVERSE, AREA_DECK, AREA_HAND

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
