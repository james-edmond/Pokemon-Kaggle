from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .cards import PAD_ROW, CardTables, card_row
from .tracker import BeliefSnapshot

MAX_TOKENS = 192
NUM_DIM = 40
KIND_SPECIAL, KIND_ENTITY, KIND_CHILD, KIND_MULTISET = 0, 1, 2, 3
OWNER_SELF, OWNER_OPP, OWNER_NEUTRAL = 0, 1, 2
N_ZONE, N_OWNER, N_KIND, N_POS = 17, 3, 4, 16
Z_GLOBAL, Z_PSUM, Z_VALUE, Z_SCRATCH = 13, 14, 15, 16
SUB_ENERGY, SUB_TOOL, SUB_PREEVO = 100, 200, 300
(F_HP, F_MAXHP, F_DMG, F_COUNT, F_POISON, F_BURN, F_ASLEEP, F_PARA, F_CONF,
 F_APPEAR, F_ACTIVE, F_BENCHIX, F_DECKN, F_HANDN, F_PRIZEN, F_TURN,
 F_SUPPORTER, F_STADIUMF, F_ENERGYATT, F_RETREATED, F_SPLIT) = range(21)
F_ENERGY0 = 21  # ..32

# AreaType values reused as zones
AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_STADIUM, AREA_LOOKING = 4, 5, 7, 12


@dataclass
class TokenizedState:
    card: np.ndarray
    numeric: np.ndarray
    owner: np.ndarray
    zone: np.ndarray
    kind: np.ndarray
    pos: np.ndarray
    mask: np.ndarray
    ref: dict = field(default_factory=dict)
    mrow: dict = field(default_factory=dict)
    n: int = 0


class _Builder:
    def __init__(self, tables: CardTables):
        self.t = tables
        self.s = TokenizedState(
            np.full(MAX_TOKENS, PAD_ROW, np.int64),
            np.zeros((MAX_TOKENS, NUM_DIM), np.float32),
            np.zeros(MAX_TOKENS, np.int64), np.zeros(MAX_TOKENS, np.int64),
            np.zeros(MAX_TOKENS, np.int64), np.zeros(MAX_TOKENS, np.int64),
            np.zeros(MAX_TOKENS, bool),
        )

    def add(self, card_id, owner, zone, kind, pos=0):
        s, i = self.s, self.s.n
        if i >= MAX_TOKENS:
            raise OverflowError("token budget exceeded")
        s.card[i] = card_row(card_id, self.t.n_rows) if card_id else PAD_ROW
        s.owner[i], s.zone[i], s.kind[i] = owner, zone, kind
        s.pos[i] = min(pos, N_POS - 1)
        s.mask[i] = True
        s.n += 1
        return i


def _add_pokemon(b, pk, owner, player_index, area, idx):
    row = b.add(pk["id"], owner, area, KIND_ENTITY, pos=idx)
    s = b.s
    s.ref[(player_index, area, idx, -1)] = row
    n = s.numeric[row]
    mx = pk.get("maxHp") or 0
    n[F_HP] = (pk.get("hp") or 0) / 300.0
    n[F_MAXHP] = mx / 300.0
    n[F_DMG] = ((mx - (pk.get("hp") or 0)) / mx) if mx else 0.0
    n[F_APPEAR] = float(pk.get("appearThisTurn") or False)
    n[F_ACTIVE] = float(area == 4)
    n[F_BENCHIX] = idx / 8.0
    for e in pk.get("energies") or []:
        if 0 <= int(e) < 12:
            n[F_ENERGY0 + int(e)] += 0.2
    for sub, zone, key in ((SUB_ENERGY, 8, "energyCards"),
                           (SUB_TOOL, 9, "tools"),
                           (SUB_PREEVO, 10, "preEvolution")):
        for j, c in enumerate(pk.get(key) or []):
            r = b.add(c["id"], owner, zone, KIND_CHILD, pos=j)
            s.ref[(player_index, area, idx, sub + j)] = r
    return row


def _visible_own_ids(obs, me):
    """Every own card id currently visible to `me`: hand, board (main + energy +
    tools + pre-evolutions, active & bench), discard, own stadium, face-up prizes."""
    cur = obs["current"]
    pl = cur["players"][me]
    ids = Counter()
    for c in pl.get("hand") or []:
        ids[c["id"]] += 1
    for area_key in ("active", "bench"):
        for pk in pl.get(area_key) or []:
            if pk is None:
                continue
            ids[pk["id"]] += 1
            for key in ("energyCards", "tools", "preEvolution"):
                for c in pk.get(key) or []:
                    ids[c["id"]] += 1
    for c in pl.get("discard") or []:
        ids[c["id"]] += 1
    for c in cur.get("stadium") or []:
        if c is not None and c.get("playerIndex") == me:
            ids[c["id"]] += 1
    for c in pl.get("prize") or []:
        if c is not None:  # face-up (revealed) prizes only
            ids[c["id"]] += 1
    return ids


def _own_union(own_deck, obs, me):
    """own deck ∪ prizes minus every visible own card id, clamped at zero."""
    union = Counter(own_deck)
    union.subtract(_visible_own_ids(obs, me))
    return Counter({cid: n for cid, n in union.items() if n > 0})


def featurize_state(obs, me, own_deck, belief, tables) -> TokenizedState:
    b = _Builder(tables)
    cur = obs["current"]
    opp = 1 - me
    players = cur["players"]
    # specials: 0 global, 1-2 player summaries, 3-4 value, 5-8 scratch
    g = b.add(0, OWNER_NEUTRAL, Z_GLOBAL, KIND_SPECIAL)
    psum_row = {}
    for pi in (me, opp):
        psum_row[pi] = b.add(
            0, OWNER_SELF if pi == me else OWNER_OPP, Z_PSUM, KIND_SPECIAL)
    for _ in range(2):
        b.add(0, OWNER_NEUTRAL, Z_VALUE, KIND_SPECIAL)
    for k in range(4):
        b.add(0, OWNER_NEUTRAL, Z_SCRATCH, KIND_SPECIAL, pos=k)
    num = b.s.numeric
    num[g, F_TURN] = cur["turn"] / 50.0
    num[g, F_SUPPORTER] = float(cur["supporterPlayed"])
    num[g, F_STADIUMF] = float(cur["stadiumPlayed"])
    num[g, F_ENERGYATT] = float(cur["energyAttached"])
    num[g, F_RETREATED] = float(cur["retreated"])

    # player summaries: deck/hand/prize counts into F_DECKN/F_HANDN/F_PRIZEN
    for pi in (me, opp):
        pl = players[pi]
        r = psum_row[pi]
        num[r, F_DECKN] = (pl.get("deckCount") or 0) / 60.0
        num[r, F_HANDN] = (pl.get("handCount") or 0) / 10.0
        num[r, F_PRIZEN] = len(pl.get("prize") or []) / 6.0

    # entities: per player active+bench via _add_pokemon (children inside)
    for pi in (me, opp):
        pl = players[pi]
        owner = OWNER_SELF if pi == me else OWNER_OPP
        active_row = None
        for i, pk in enumerate(pl.get("active") or []):
            if pk is None:
                continue
            active_row = _add_pokemon(b, pk, owner, pi, AREA_ACTIVE, i)
        for i, pk in enumerate(pl.get("bench") or []):
            if pk is None:
                continue
            _add_pokemon(b, pk, owner, pi, AREA_BENCH, i)
        # special conditions apply to this player's active Pokémon
        if active_row is not None:
            n = num[active_row]
            n[F_POISON] = float(pl.get("poisoned") or False)
            n[F_BURN] = float(pl.get("burned") or False)
            n[F_ASLEEP] = float(pl.get("asleep") or False)
            n[F_PARA] = float(pl.get("paralyzed") or False)
            n[F_CONF] = float(pl.get("confused") or False)

    # stadium: entity token(s), owner by the card's playerIndex vs me
    for i, c in enumerate(cur.get("stadium") or []):
        if c is None:
            continue
        pidx = c.get("playerIndex")
        owner = OWNER_SELF if pidx == me else OWNER_OPP
        row = b.add(c["id"], owner, AREA_STADIUM, KIND_ENTITY, pos=i)
        b.s.ref[(pidx, AREA_STADIUM, i, -1)] = row

    # own hand tokens: one entity per card
    for i, c in enumerate(players[me].get("hand") or []):
        if c is None:
            continue
        row = b.add(c["id"], OWNER_SELF, AREA_HAND, KIND_ENTITY, pos=i)
        b.s.ref[(me, AREA_HAND, i, -1)] = row

    # looking tokens: entity tokens, owner self
    looking = cur.get("looking")
    if looking is not None:
        for i, c in enumerate(looking):
            if c is None:
                continue
            row = b.add(c["id"], OWNER_SELF, AREA_LOOKING, KIND_ENTITY, pos=i)
            b.s.ref[(me, AREA_LOOKING, i, -1)] = row

    # ---- multisets (droppable under truncation) --------------------------------
    # own deck∪prizes union
    union = _own_union(own_deck, obs, me)
    # own & opp discard
    own_disc = Counter(c["id"] for c in players[me].get("discard") or [])
    opp_disc = Counter(c["id"] for c in players[opp].get("discard") or [])
    # belief multisets. opp_deck and opp_hidden_pool both live at zone DECK / owner
    # OPP; merge colliding card ids into one token with F_SPLIT = pool fraction.
    opp_hand = Counter(belief.opp_hand)
    opp_deck_raw = Counter(belief.opp_deck)
    opp_pool_raw = Counter(belief.opp_hidden_pool)
    opp_deck_merged = Counter()
    deck_split = {}
    for cid in set(opp_deck_raw) | set(opp_pool_raw):
        dk = opp_deck_raw.get(cid, 0)
        po = opp_pool_raw.get(cid, 0)
        total = dk + po
        opp_deck_merged[cid] = total
        deck_split[cid] = (po / total) if total else 0.0

    # groups: (owner, zone, Counter, feat_slot, scale, split_map)
    groups = [
        (OWNER_SELF, AREA_DECK, union, F_COUNT, 0.25, None),
        (OWNER_SELF, AREA_DISCARD, own_disc, F_COUNT, 0.25, None),
        (OWNER_OPP, AREA_DISCARD, opp_disc, F_COUNT, 0.25, None),
        (OWNER_OPP, AREA_HAND, opp_hand, F_COUNT, 0.25, None),
        (OWNER_OPP, AREA_DECK, opp_deck_merged, F_COUNT, 0.25, deck_split),
    ]

    # flatten to a droppable pool of rows, smallest count dropped first
    all_rows = []  # (count, owner, zone, cid, feat_slot, scale, split_val)
    for owner, zone, counts, slot, scale, split in groups:
        for cid, cnt in counts.items():
            sv = split.get(cid, 0.0) if split is not None else None
            all_rows.append((cnt, owner, zone, cid, slot, scale, sv))

    total_ms = len(all_rows)
    fixed_n = b.s.n  # specials + entities + children (never dropped)
    budget = MAX_TOKENS - fixed_n
    if budget < 0:
        raise AssertionError(
            f"non-droppable tokens ({fixed_n}) exceed MAX_TOKENS ({MAX_TOKENS})")

    if total_ms > budget:
        # keep the largest-count rows; drop smallest first
        all_rows.sort(key=lambda r: r[0])  # ascending by count
        kept = all_rows[total_ms - budget:]
        dropped = total_ms - budget
        num[g, F_SPLIT] = dropped / total_ms if total_ms else 0.0
    else:
        kept = all_rows

    s = b.s
    for cnt, owner, zone, cid, slot, scale, sv in kept:
        row = b.add(cid, owner, zone, KIND_MULTISET)
        s.mrow[(owner, zone, cid)] = row
        s.numeric[row, slot] = cnt * scale
        if sv is not None:
            s.numeric[row, F_SPLIT] = sv

    return b.s
