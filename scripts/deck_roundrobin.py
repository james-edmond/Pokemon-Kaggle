"""Pick the submission deck: play the generalist deck-vs-deck across the portfolio and
report each deck's average win rate across the field. Run from repo root (base python),
one engine process at a time. Usage: python scripts/deck_roundrobin.py [games_per_pair]"""
import os
import sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ptcg.actors import play_versus
from ptcg.cards import build_tables
from ptcg.decks import all_decks, deck as get_deck
from ptcg.model import PolicyModel, student_config

CKPT = os.path.join(REPO, "submission_src", "policy.pt")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tables = build_tables()
    model = PolicyModel(student_config(tables))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    names = all_decks()
    wins = {a: 0 for a in names}
    games = {a: 0 for a in names}
    seed = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = get_deck(a), get_deck(b)
            for g in range(n):
                seed += 1
                gen = torch.Generator().manual_seed(seed)
                # deck a on seat g%2, deck b on the other; play_versus returns 1 iff seat0-model won
                seat = g % 2
                decks = (da, db) if seat == 0 else (db, da)
                with torch.no_grad():
                    r = play_versus(model, model, tables, decks, gen, model_seat=seat)
                # r==1 means the seat-`seat` player (deck a) won
                wins[a] += r
                wins[b] += (1 - r)
                games[a] += 1
                games[b] += 1
    rows = sorted(((wins[a] / games[a], a) for a in names), reverse=True)
    print(f"{'deck':<26}{'winrate':>9}{'games':>8}")
    for wr, a in rows:
        print(f"{a:<26}{wr:>9.3f}{games[a]:>8}")
    print(f"\nBEST: {rows[0][1]}  ({rows[0][0]:.3f})")


if __name__ == "__main__":
    main()
