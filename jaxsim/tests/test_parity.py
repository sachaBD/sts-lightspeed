"""Parity against the C++ simulator: random decks, random legal play, every step compared.

Needs the oracle binary (tools/build_oracle.sh); skipped otherwise. JAX consumes the C++ random outcomes
(drawn cards, Champ rolls) through the private forced path, so states must match exactly.
"""
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsim
from jaxsim._core import init as core_init, step as core_step
from jaxsim._core.rand import Forced
from tests.oracle import convert
from tests.oracle.client import BINARY, Oracle

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="oracle not built (tools/build_oracle.sh)")

SPEC = jaxsim.DEFAULT_SPEC
FIGHTS = 200
_init = jax.jit(lambda start, key, forced: core_init.init(SPEC, start, key, forced))
_step = jax.jit(lambda s, a, key, forced: core_step.step(SPEC, s, a, key, forced))
KEY = jax.random.key(0)


def test_card_table_matches_cpp():
    """cost / type / exhausts / ethereal of every supported row equal the C++ card data."""
    from jaxsim._core.tables import cards as tc, ids
    cpp_types = ("ATTACK", "SKILL", "POWER", "CURSE", "STATUS")
    jax_types = ("ATTACK", "SKILL", "POWER", "STATUS", "CURSE")
    oracle = Oracle()
    try:
        cpp = {(ids.CARDS_CPP[c[0]], c[1]): c[2:] for c in oracle.cards()}
    finally:
        oracle.close()
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


def random_deck(rng: random.Random):
    from jaxsim._core.tables import cards as tc
    pool = [c for c in jaxsim.supported_cards() if c not in tc.NOT_IN_DECKS + ("ASCENDERS_BANE",)]
    deck = [("ASCENDERS_BANE", 0)] + [(c, 0) for c in ["STRIKE_RED"] * 4 + ["DEFEND_RED"] * 4 + ["BASH"]]
    for _ in range(rng.randint(0, 14)):
        card, up = rng.choice(pool), rng.randint(0, 1)
        deck.append((card, rng.randint(1, 6) if card == "SEARING_BLOW" and up else up))
    if rng.random() < 0.3:
        deck.append(("WOUND", 0))
    return deck


def choose(rng, fight, legal, obs):
    """Even fights: uniform random. Odd fights: play cards before ending the turn, preferring attacks and
    strong decks, so fights reach the Champ's phase 2 and wins."""
    if fight % 2 == 0 or len(legal) == 1:
        return rng.choice(sorted(legal))
    cards = [a for a in legal if a != SPEC.END_TURN]
    return rng.choice(cards)


def fight_setup(rng, fight):
    deck, hp = random_deck(rng), rng.randint(30, 80)
    if fight % 2:
        deck += [(c, 1) for c in ["INFLAME", "TWIN_STRIKE", "POMMEL_STRIKE", "SHRUG_IT_OFF"] * rng.randint(1, 3)]
        hp = 300  # long fights: reach phase 2, Execute and wins
    return deck, hp


def created_in_hand(obs, a, hand_size):
    """Cards the played card puts straight into the hand (not draws), per slot."""
    from jaxsim._core.tables import cards as tc
    row = tc.TABLE[int(obs.slot_card[a]), min(int(obs.slot_upgraded[a]), 1)]
    out = np.zeros(SPEC.D, np.int32)
    if row[tc.col("create_pile")] == tc.TO_HAND:
        card = int(row[tc.col("create_card")])
        hit = np.flatnonzero((np.asarray(obs.slot_card) == card) & (np.asarray(obs.slot_upgraded) == 0))
        slot = int(hit[0]) if len(hit) else int(np.flatnonzero(np.asarray(obs.slot_card) == 0)[0])
        out[slot] = min(int(row[tc.col("create_n")]), max(SPEC.H - int(hand_size), 0))
    return out


def forced(draw, roll):
    return Forced(draw=jnp.asarray(draw, jnp.int32), roll=jnp.int32(roll))


def roll_of(resp):
    rolls = resp["ai_rolls"]
    assert len(rolls) <= 1, rolls
    return rolls[0] if rolls else 0


def check(resp, s, where):
    obs, out = jaxsim.make_env().observe(s), jaxsim.make_env().outcome(s)
    bad = convert.diff(convert.expected(resp["state"], obs, SPEC.D), convert.actual(obs, out))
    assert not bad, f"{where}:\n  " + "\n  ".join(bad)
    assert int(out.error) == 0, f"{where}: error bits {int(out.error)}"
    if not out.done:
        cpp = set(convert.legal_actions(resp["actions"], obs, SPEC))
        jax_legal = set(np.flatnonzero(np.asarray(jaxsim.make_env().legal_mask(s))).tolist())
        assert cpp == jax_legal, f"{where}: legal c++ {sorted(cpp)} != jax {sorted(jax_legal)}"


@pytest.mark.parametrize("fight", range(FIGHTS))
def test_random_fight(fight):
    rng = random.Random(fight)
    deck, hp = fight_setup(rng, fight)
    oracle = Oracle()
    try:
        resp = oracle.start(seed=1000 + fight, hp=hp, max_hp=max(hp, 80), deck=convert.cpp_deck(deck))
        start = jaxsim.make_start(deck, hp=hp, max_hp=max(hp, 80))
        obs0 = jaxsim.make_env().observe(core_init.init(SPEC, start, KEY))  # slot table only
        hand = convert.slot_counts(obs0, resp["state"]["hand"], SPEC.D)
        s = _init(start, KEY, forced(hand, roll_of(resp)))
        check(resp, s, f"fight {fight} init")
        for i in range(2000):
            obs = jaxsim.make_env().observe(s)
            if jaxsim.make_env().outcome(s).done:
                break
            legal = convert.legal_actions(resp["actions"], obs, SPEC)
            a = choose(rng, fight, legal, obs)
            before = np.asarray(obs.piles)[1].astype(np.int32)
            resp = oracle.step(legal[a])
            after = convert.slot_counts(obs, resp["state"]["hand"], SPEC.D, new={}, ignore_cost=True)
            if a == SPEC.END_TURN:
                drawn = after
            else:
                rest = before.copy()
                rest[a] -= 1
                drawn = after - rest
                drawn -= created_in_hand(obs, a, rest.sum())
            s = _step(s, jnp.int32(a), KEY, forced(np.maximum(drawn, 0), roll_of(resp)))
            check(resp, s, f"fight {fight} step {i} action {a}")
    finally:
        oracle.close()
