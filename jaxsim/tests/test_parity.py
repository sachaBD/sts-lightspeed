"""Exact parity against the C++ simulator.

- `test_card_table_matches_cpp` (gate): card table cost / type / exhausts / ethereal equal the C++ card data.
- `test_random_fight` (diagnostic, opt in with PARITY_FIGHTS=N): random decks and random legal play, every step
  compared exactly on the forced path. Useful for hunting differences; not a gate.
"""
import os
import random

import pytest

import jaxsim
from tests.oracle import decks
from tests.oracle.harness import SPEC, Fight

FIGHTS = int(os.environ.get("PARITY_FIGHTS", "0"))


def test_card_table_matches_cpp(oracle):
    """cost / type / exhausts / ethereal of every supported row equal the C++ card data."""
    from jaxsim._core.tables import cards as tc, ids
    cpp_types = ("ATTACK", "SKILL", "POWER", "CURSE", "STATUS")
    jax_types = ("ATTACK", "SKILL", "POWER", "STATUS", "CURSE")
    cpp = {(ids.CARDS_CPP[c[0]], c[1]): c[2:] for c in oracle.cards()}
    bad = []
    for name in tc.SUPPORTED_NAMES:
        for up in (0, 1):
            row = tc.TABLE[jaxsim.CARD_NAMES.index(name), up]
            cost, ctype, exhausts, ethereal = cpp[(name, up)]
            mine = (int(row[tc.col("cost")]), jax_types[row[tc.col("type")]], bool(row[tc.col("exhausts")]),
                    bool(row[tc.col("ethereal")]))
            theirs = (cost, cpp_types[ctype], bool(exhausts), bool(ethereal))
            if mine != theirs:
                bad.append(f"{name}+{up}: jaxsim {mine} != c++ {theirs}")
    assert not bad, "\n".join(bad)


def choose(rng, fight, legal):
    """Even fights: uniform random. Odd fights: play cards before ending the turn, so fights reach the Champ's
    phase 2 and wins."""
    legal = sorted(legal)
    if (fight % 2 == 0 and fight % 3 != 2) or len(legal) == 1:
        return rng.choice(legal)
    return rng.choice([a for a in legal if a != SPEC.END_TURN])


@pytest.mark.skipif(FIGHTS == 0, reason="diagnostic: PARITY_FIGHTS=N to run")
@pytest.mark.parametrize("fight", range(max(FIGHTS, 1)))
def test_random_fight(fight, oracle):
    rng = random.Random(fight)
    deck, hp = decks.fight_setup(rng, fight)
    relics = decks.random_relics(random.Random(10_000 + fight))
    pots = decks.random_potions(random.Random(20_000 + fight))
    f = Fight(oracle, f"fight {fight}", deck, hp, relics=relics, potions=pots, seed=1000 + fight)
    for _ in range(2000):
        if f.done:
            break
        f.act(choose(rng, fight, f.legal()))
