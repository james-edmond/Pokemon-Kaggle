import numpy as np
from ptcg.cards import (
    ATTR_DIM, PAD_ROW, UNK_ROW, build_tables, card_row,
    A_HP, A_BASIC, A_STAGE1, A_STAGE2, A_MEGA, A_IS_BASIC_ENERGY,
)


def test_tables_shape_and_reserved_rows():
    t = build_tables()
    assert t.attr.shape == (t.n_rows, ATTR_DIM)
    assert t.attr.dtype == np.float32
    assert np.all(t.attr[PAD_ROW] == 0) and np.all(t.attr[UNK_ROW] == 0)


def test_gardevoir_line_stages():
    t = build_tables()
    r745, r746, r747 = (t.attr[card_row(c, t.n_rows)] for c in (745, 746, 747))
    assert r745[A_BASIC] == 1 and r745[A_STAGE1] == 0
    assert r746[A_STAGE1] == 1
    assert r747[A_STAGE2] == 1 and r747[A_MEGA] == 1 and r747[A_HP] > 0


def test_basic_energy_flag_and_unknown_id():
    t = build_tables()
    assert t.attr[card_row(3, t.n_rows)][A_IS_BASIC_ENERGY] == 1  # Basic Water
    assert card_row(999999, t.n_rows) == UNK_ROW
