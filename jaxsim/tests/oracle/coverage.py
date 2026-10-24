"""Coverage of the parity fights' decks and policies, measured on the C++ side only: outcomes, Champ moves and
plays per card. Run: uv run python -m tests.oracle.coverage"""
import collections
import os
import random

from jaxsim._core.tables import ids
from tests.oracle import convert, decks
from tests.oracle.client import Oracle


def main():
    moves, outcomes, phase2, played = (collections.Counter(), collections.Counter(), 0, collections.Counter())
    for fight in range(int(os.environ.get("PARITY_FIGHTS", "300"))):
        rng = random.Random(fight)
        deck, hp = decks.fight_setup(rng, fight)
        o = Oracle()
        r = o.start(seed=1000 + fight, hp=hp, max_hp=max(hp, 80), deck=convert.cpp_deck(deck))
        p2 = False
        while not r["state"]["outcome"]:
            st = r["state"]
            moves[ids.MOVES_CPP[st["enemy"]["move"]]] += 1
            p2 |= bool(st["enemy"]["misc"] & 4)
            cards = [a for a in r["actions"] if a["type"] == convert.ACTION_CARD]
            end = [a for a in r["actions"] if a["type"] == convert.ACTION_END_TURN]
            pool = cards if ((fight % 2 or fight % 3 == 2) and cards) else cards + end
            a = rng.choice(pool)
            if a["type"] == convert.ACTION_CARD:
                played[ids.CARDS_CPP[a["card"][0]]] += 1
            r = o.step(a["bits"])
        outcomes[{1: "win", 2: "loss"}[r["state"]["outcome"]]] += 1
        phase2 += p2
        o.close()
    print(dict(outcomes), "fights reaching phase 2:", phase2)
    print(dict(moves))
    print("plays per card:", dict(sorted(played.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
