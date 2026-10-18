"""Coverage of the parity fights: outcomes and Champ moves seen. Run: uv run python -m tests.oracle.coverage"""
import collections
import random

import jaxsim
from jaxsim._core.tables import ids
from tests import test_parity as T
from tests.oracle import convert
from tests.oracle.client import Oracle


def main():
    moves, outcomes, phase2, played = collections.Counter(), collections.Counter(), 0, collections.Counter()
    for fight in range(T.FIGHTS):
        rng = random.Random(fight)
        deck, hp = T.fight_setup(rng, fight)
        o = Oracle()
        r = o.start(seed=1000 + fight, hp=hp, max_hp=max(hp, 80), deck=convert.cpp_deck(deck))
        env = jaxsim.make_env()
        obs = env.observe(env.init(jaxsim.make_start(deck, hp=hp, max_hp=max(hp, 80)), T.KEY))
        p2 = False
        while not r["state"]["outcome"]:
            st = r["state"]
            moves[ids.MOVES_CPP[st["enemy"]["move"]]] += 1
            p2 |= bool(st["enemy"]["misc"] & 4)
            legal = convert.legal_actions(r["actions"], obs, T.SPEC)
            a = T.choose(rng, fight, legal, obs)
            if a != T.SPEC.END_TURN:
                played[jaxsim.CARD_NAMES[int(obs.slot_card[a])]] += 1
            r = o.step(legal[a])
            obs = None if r["state"]["outcome"] else obs
        outcomes[{1: "win", 2: "loss"}[r["state"]["outcome"]]] += 1
        phase2 += p2
        o.close()
    print(dict(outcomes), "fights reaching phase 2:", phase2)
    print(dict(moves))
    print("plays per card:", dict(sorted(played.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
