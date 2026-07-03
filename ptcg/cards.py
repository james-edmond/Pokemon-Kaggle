from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .engine import _load_game

PAD_ROW, UNK_ROW, N_RESERVED = 0, 1, 2
HEADROOM = 256
N_ENERGY_TYPES = 12  # EnergyType 0..11

# attr slots
A_HP, A_RETREAT, A_BASIC, A_STAGE1, A_STAGE2, A_EX, A_MEGA, A_TERA, A_ACESPEC = range(9)
A_IS_POKEMON, A_IS_ITEM, A_IS_TOOL, A_IS_SUPPORTER, A_IS_STADIUM = range(9, 14)
A_IS_BASIC_ENERGY, A_IS_SPECIAL_ENERGY = 14, 15
A_ENERGY_TYPE0 = 16  # 16..27 one-hot by EnergyType
ATTR_DIM = A_ENERGY_TYPE0 + N_ENERGY_TYPES  # 28

K_DAMAGE, K_COST_TOTAL = 0, 1
K_COST_TYPE0 = 2  # 2..13 cost count by EnergyType
ATK_DIM = K_COST_TYPE0 + N_ENERGY_TYPES  # 14


@dataclass(frozen=True)
class CardTables:
    n_rows: int
    attr: np.ndarray
    n_attack_rows: int
    attack_feat: np.ndarray
    _attack_index: dict


def card_row(card_id: int, n_rows: int) -> int:
    r = card_id + N_RESERVED
    return r if 0 <= card_id and r < n_rows else UNK_ROW


def attack_row(attack_id: int, tables: CardTables) -> int:
    return tables._attack_index.get(attack_id, UNK_ROW)


@lru_cache(maxsize=1)
def build_tables() -> CardTables:
    _load_game()  # ensures cg is importable
    from cg import api

    cards = api.all_card_data()
    attacks = api.all_attack()

    n_rows = N_RESERVED + max(c.cardId for c in cards) + HEADROOM
    attr = np.zeros((n_rows, ATTR_DIM), dtype=np.float32)
    for c in cards:
        v = attr[card_row(c.cardId, n_rows)]
        v[A_HP] = (c.hp or 0) / 300.0
        v[A_RETREAT] = (c.retreatCost or 0) / 5.0
        v[A_BASIC], v[A_STAGE1], v[A_STAGE2] = float(c.basic), float(c.stage1), float(c.stage2)
        v[A_EX], v[A_MEGA], v[A_TERA], v[A_ACESPEC] = (
            float(c.ex), float(c.megaEx), float(c.tera), float(c.aceSpec)
        )
        ct = int(c.cardType)
        v[A_IS_POKEMON] = float(ct == 0)
        v[A_IS_ITEM] = float(ct == 1)
        v[A_IS_TOOL] = float(ct == 2)
        v[A_IS_SUPPORTER] = float(ct == 3)
        v[A_IS_STADIUM] = float(ct == 4)
        v[A_IS_BASIC_ENERGY] = float(ct == 5)
        v[A_IS_SPECIAL_ENERGY] = float(ct == 6)
        et = int(c.energyType)
        if 0 <= et < N_ENERGY_TYPES:
            v[A_ENERGY_TYPE0 + et] = 1.0

    n_attack_rows = N_RESERVED + len(attacks)
    attack_feat = np.zeros((n_attack_rows, ATK_DIM), dtype=np.float32)
    attack_index = {}
    for i, a in enumerate(attacks):
        row = N_RESERVED + i
        attack_index[a.attackId] = row
        attack_feat[row, K_DAMAGE] = (a.damage or 0) / 300.0
        attack_feat[row, K_COST_TOTAL] = len(a.energies) / 5.0
        for e in a.energies:
            if 0 <= int(e) < N_ENERGY_TYPES:
                attack_feat[row, K_COST_TYPE0 + int(e)] += 1.0 / 5.0

    return CardTables(n_rows, attr, n_attack_rows, attack_feat, attack_index)
