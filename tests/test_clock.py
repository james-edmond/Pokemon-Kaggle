from ptcg.clock import SearchClock, forced_picks


def _sel(n, lo, hi, types=None):
    opts = [{"type": t} for t in (types or [0] * n)]
    return {"option": opts, "minCount": lo, "maxCount": hi}


def test_forced_picks_single_required_option():
    assert forced_picks(_sel(1, 1, 1)) == [0]


def test_forced_picks_take_all():
    assert forced_picks(_sel(3, 3, 3)) == [0, 1, 2]


def test_forced_picks_real_choices_return_none():
    assert forced_picks(_sel(1, 0, 1)) is None      # [] vs [0] is a real choice
    assert forced_picks(_sel(3, 1, 1)) is None
    assert forced_picks(_sel(3, 2, 2)) is None      # choose WHICH 2 of 3
    assert forced_picks({"option": []}) is None     # malformed: no crash


def test_forced_picks_defers_order_semantic_take_alls():
    # SKILL options (type 15): order is the decision -> not forced
    sel = {"option": [{"type": 15}, {"type": 15}], "minCount": 2, "maxCount": 2,
           "context": 34}
    assert forced_picks(sel) is None
    # TO_DECK_BOTTOM context: pick sequence may encode placement -> not forced
    sel2 = {"option": [{"type": 3}] * 3, "minCount": 3, "maxCount": 3,
            "context": 10}
    assert forced_picks(sel2) is None
    sel3 = {"option": [{"type": 3}] * 3, "minCount": 3, "maxCount": 3,
            "context": 9}
    assert forced_picks(sel3) is None
    # plain take-all (e.g. DISCARD context) still forced
    sel4 = {"option": [{"type": 3}] * 3, "minCount": 3, "maxCount": 3,
            "context": 8}
    assert forced_picks(sel4) == [0, 1, 2]
    # missing context key: treated as order-safe (backward compatible)
    sel5 = {"option": [{"type": 3}] * 3, "minCount": 3, "maxCount": 3}
    assert forced_picks(sel5) == [0, 1, 2]


def test_slice_basics_and_importance():
    c = SearchClock(bank_s=480.0, floor_s=60.0, cap_s=20.0, expected_total_moves=80)
    base = c.slice_for(_sel(3, 1, 1))
    assert abs(base - 6.0) < 1e-9                   # 480/80
    atk = c.slice_for(_sel(3, 1, 1, types=[0, 13, 0]))   # ATTACK option present
    assert abs(atk - 9.0) < 1e-9                    # 1.5x importance
    many = c.slice_for(_sel(6, 1, 1))
    assert abs(many - 9.0) < 1e-9                   # >=6 options -> 1.5x
    assert c.slice_for(_sel(1, 1, 1)) == 0.0        # trivial -> no search


def test_slice_floor_and_cap():
    c = SearchClock(bank_s=480.0, floor_s=60.0, cap_s=20.0, expected_total_moves=80)
    c.charge(425.0)
    assert c.remaining == 55.0
    assert c.slice_for(_sel(3, 1, 1)) == 0.0        # below floor -> search off
    c2 = SearchClock(bank_s=480.0, cap_s=5.0)
    assert c2.slice_for(_sel(3, 1, 1)) == 5.0       # cap binds (480/80=6 > 5)


def test_bank_is_process_lifetime_and_moves_reset():
    c = SearchClock()
    c.charge(100.0)
    for _ in range(30):
        c.note_move()
    assert c.moves_this_game == 30
    c.new_game()
    assert c.moves_this_game == 0
    assert c.spent == 100.0                          # never resets


def test_expected_remaining_moves_floor():
    c = SearchClock(bank_s=480.0, expected_total_moves=80)
    for _ in range(75):
        c.note_move()
    # exp_rem = max(20, 80-75) = 20 -> slice = 480/20 = 24 -> capped at 20
    assert c.slice_for(_sel(3, 1, 1)) == 20.0
