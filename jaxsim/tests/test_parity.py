"""Parity against the C++ simulator: random decks, random legal play, every step compared.

Needs the oracle binary (tools/build_oracle.sh); skipped otherwise. JAX consumes the C++ random outcomes
(drawn cards, Champ rolls) through the private forced path, so states must match exactly.
"""
import os
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
# Default: 60 fights (all three deck types, ~15 s). PARITY_FIGHTS=300 for the full run before big merges.
FIGHTS = int(os.environ.get("PARITY_FIGHTS", "60"))
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
    if (fight % 2 == 0 and fight % 3 != 2) or len(legal) == 1:
        return rng.choice(sorted(legal))
    cards = [a for a in legal if a != SPEC.END_TURN]
    return rng.choice(cards)


POWER_MIX = ["METALLICIZE", "DEMON_FORM", "BARRICADE", "BERSERK", "BRUTALITY", "COMBUST", "CORRUPTION",
             "DARK_EMBRACE", "EVOLVE", "FEEL_NO_PAIN", "FIRE_BREATHING", "JUGGERNAUT", "RUPTURE", "RAGE",
             "FLAME_BARRIER"]
HOOK_FUEL = ["SECOND_WIND", "FIEND_FIRE", "BURNING_PACT", "TRUE_GRIT", "EXHUME", "ARMAMENTS", "WARCRY", "HEADBUTT", "SEVER_SOUL", "GHOSTLY_ARMOR", "CARNAGE", "WILD_STRIKE", "POWER_THROUGH",
             "IMMOLATE", "RECKLESS_CHARGE", "OFFERING", "HEMOKINESIS", "SENTINEL", "SHRUG_IT_OFF", "BLOOD_FOR_BLOOD"]


def fight_setup(rng, fight):
    deck, hp = random_deck(rng), rng.randint(30, 80)
    if fight % 3 == 2:  # powers and the cards that trigger them, long fights
        deck += [(rng.choice(POWER_MIX), rng.randint(0, 1)) for _ in range(rng.randint(2, 5))]
        deck += [(rng.choice(HOOK_FUEL), rng.randint(0, 1)) for _ in range(rng.randint(2, 6))]
        return deck, 250
    if fight % 2:
        deck += [(c, 1) for c in ["INFLAME", "TWIN_STRIKE", "POMMEL_STRIKE", "SHRUG_IT_OFF"] * rng.randint(1, 3)]
        hp = 300  # long fights: reach phase 2, Execute and wins
    return deck, hp


def forced(draw, roll, exhaust=None):
    exhaust = np.zeros(SPEC.D, np.int32) if exhaust is None else exhaust
    return Forced(draw=jnp.asarray(draw, jnp.int32), roll=jnp.int32(roll), exhaust=jnp.asarray(exhaust, jnp.int32))


def roll_of(resp):
    rolls = resp["ai_rolls"]
    assert len(rolls) <= 1, rolls
    return rolls[0] if rolls else 0


def check(resp, s, where):
    obs, out = jaxsim.make_env().observe(s), jaxsim.make_env().outcome(s)
    try:
        exp = convert.expected(resp["state"], obs, SPEC.D)
    except KeyError as e:
        slots = [f"{d}:{jaxsim.CARD_NAMES[int(obs.slot_card[d])]}+{int(obs.slot_upgraded[d])}/c{int(obs.slot_cost[d])}"
                 f"[{','.join(str(int(x)) for x in np.asarray(obs.piles)[:4, d])}]"
                 for d in range(SPEC.D) if int(obs.slot_card[d])]
        st = resp["state"]
        cpp = {k: [c for c in st[k]] for k in ("draw", "hand", "discard", "exhaust")}
        raise KeyError(f"{where}: {e}; jax slots: {' '.join(slots)}; c++ piles {cpp}") from None
    act = convert.actual(obs, out)
    if exp["done"]:  # after the fight ends only the result matters (leftover queued effects are not modelled)
        exp = {"done": exp["done"], "won": exp["won"], "player": exp["player"][:2]}
        act = {k: act[k] for k in ("done", "won")} | {"player": act["player"][:2]}
    bad = convert.diff(exp, act)
    if bad:
        slots = [f"{d}:{jaxsim.CARD_NAMES[int(obs.slot_card[d])]}+{int(obs.slot_upgraded[d])}"
                 f"/c{int(obs.slot_cost[d])}/m{int(obs.slot_misc[d])}" for d in range(SPEC.D) if int(obs.slot_card[d])]
        bad.append("slots: " + " ".join(slots))
        bad.append("c++ player: " + str(resp["state"]["player"]) + f" jax known_top {np.asarray(obs.known_top).tolist()}")
        from jaxsim._core.tables import ids
        nm = lambda cs: [f"{ids.CARDS_CPP[c[0]]}+{c[1]}" for c in cs]
        bad.append(f"c++ drawn: {nm(resp.get('drawn', []))} exhausted: {nm(resp.get('exhausted', []))}")
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
        hand = convert.slot_counts(obs0, resp["drawn"], SPEC.D)
        s = _init(start, KEY, forced(hand, roll_of(resp)))
        check(resp, s, f"fight {fight} init")
        for i in range(2000):
            obs = jaxsim.make_env().observe(s)
            if jaxsim.make_env().outcome(s).done:
                break
            try:
                legal = convert.legal_actions(resp["actions"], obs, SPEC)
            except KeyError as e:
                slots = [f"{d}:{jaxsim.CARD_NAMES[int(obs.slot_card[d])]}+{int(obs.slot_upgraded[d])}/c{int(obs.slot_cost[d])}"
                         f" h{int(obs.piles[1, d])}d{int(obs.piles[0, d])}x{int(obs.piles[2, d])}e{int(obs.piles[3, d])}"
                         for d in range(SPEC.D) if int(obs.slot_card[d])]
                raise KeyError(f"fight {fight} step {i}: {e}; jax slots: {' '.join(slots)}; "
                               f"c++ hand {[c for c in resp['state']['hand']]}") from None
            a = choose(rng, fight, legal, obs)
            if os.environ.get("PARITY_DEBUG") and resp["state"]["input_state"] == 2:
                print("SELECT", resp["state"]["select_task"], [(x["bits"], x.get("select_idx"), x.get("card")) for x in resp["actions"]],
                      "hand", [c[:2] for c in resp["state"]["hand"]], "chosen", a, legal[a])
            resp = oracle.step(legal[a])
            if os.environ.get("PARITY_DEBUG"):
                from jaxsim._core.tables import ids
                print("  after", a, "c++ hand", [ids.CARDS_CPP[c[0]] for c in resp["state"]["hand"]],
                      "top of draw", [ids.CARDS_CPP[c[0]] for c in resp["state"]["draw"][-2:]], "input", resp["state"]["input_state"],
                      "c++ enemy", resp["state"]["enemy"], "outcome", resp["state"]["outcome"])
            new = {}
            drawn = convert.slot_counts(obs, resp["drawn"], SPEC.D, new=new, ignore_cost=True)
            exhausted = convert.slot_counts(obs, resp["exhausted"], SPEC.D, new=new, ignore_cost=True)
            if os.environ.get("PARITY_DEBUG"):
                print("  pre-step jax known_top", np.asarray(obs.known_top).tolist(), "forced drawn", np.flatnonzero(drawn).tolist(),
                      "jax enemy", np.asarray(obs.enemy).tolist(), "player", np.asarray(obs.player).tolist())
            s = _step(s, jnp.int32(a), KEY, forced(drawn, roll_of(resp), exhausted))
            check(resp, s, f"fight {fight} step {i} action {a}")
    finally:
        oracle.close()
