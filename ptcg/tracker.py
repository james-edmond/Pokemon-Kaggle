from collections import Counter
from dataclasses import dataclass

LOG_MOVE_CARD = 6
LOG_MOVE_CARD_REVERSE = 7
AREA_DECK = 1
AREA_HAND = 2
_HIDDEN = (AREA_DECK, AREA_HAND)


@dataclass(frozen=True)
class BeliefSnapshot:
    opp_hand: dict
    opp_deck: dict
    opp_hidden_pool: dict


class BeliefTracker:
    """Membership-only knowledge of the opponent's hidden zones, from one seat's logs."""

    def __init__(self, my_index: int):
        self.me = my_index
        self._hand = Counter()
        self._deck = Counter()
        self._pool = Counter()

    def _zone(self, area):
        return self._hand if area == AREA_HAND else self._deck

    def update(self, logs: list) -> None:
        for lg in logs or []:
            if lg.get("playerIndex") != 1 - self.me:
                continue
            t = lg.get("type")
            fr, to = lg.get("fromArea"), lg.get("toArea")
            if t == LOG_MOVE_CARD:
                cid = lg.get("cardId")
                if cid is None:
                    continue
                if fr in _HIDDEN:
                    z = self._zone(fr)
                    if z[cid] > 0:
                        z[cid] -= 1
                    elif self._pool[cid] > 0:
                        self._pool[cid] -= 1
                if to in _HIDDEN:
                    self._zone(to)[cid] += 1
            elif t == LOG_MOVE_CARD_REVERSE:
                if fr in _HIDDEN and to in _HIDDEN:
                    self._pool.update(self._zone(fr))
                    self._zone(fr).clear()

    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            {k: v for k, v in self._hand.items() if v > 0},
            {k: v for k, v in self._deck.items() if v > 0},
            {k: v for k, v in self._pool.items() if v > 0},
        )
