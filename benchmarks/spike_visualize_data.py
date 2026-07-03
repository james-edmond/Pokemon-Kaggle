"""Probe: does VisualizeData expose deck order / prize identity? Run once, read output."""
import json
import random
from ptcg.engine import BattleSession, load_sample_deck, random_picks

deck = load_sample_deck()
s = BattleSession(deck, list(deck))
# cg is only on sys.path after the engine loads (BattleSession -> _load_game)
from cg.game import visualize_data
rng = random.Random(0)
for _ in range(10):
    if s.done:
        break
    s.select(random_picks(s.obs, rng))
d = json.loads(visualize_data())
print(json.dumps(d, indent=1)[:4000])
s.close()
