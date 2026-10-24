"""Deck, relic and potion generators shared by the parity diagnostics, the statistical parity and the invariants."""
import random

import jaxsim

POWER_MIX = ["METALLICIZE", "DEMON_FORM", "BARRICADE", "BERSERK", "BRUTALITY", "COMBUST", "CORRUPTION",
             "DARK_EMBRACE", "FEEL_NO_PAIN", "FIRE_BREATHING", "JUGGERNAUT", "RUPTURE", "RAGE",
             "FLAME_BARRIER"]
HOOK_FUEL = ["SECOND_WIND", "FIEND_FIRE", "BURNING_PACT", "TRUE_GRIT", "EXHUME", "ARMAMENTS", "WARCRY", "HEADBUTT",
             "SEVER_SOUL", "GHOSTLY_ARMOR", "CARNAGE", "WILD_STRIKE", "POWER_THROUGH", "IMMOLATE", "RECKLESS_CHARGE",
             "OFFERING", "HEMOKINESIS", "SENTINEL", "SHRUG_IT_OFF", "BLOOD_FOR_BLOOD"]
# run data ranges for relics whose StartSpec.relic_data matters
RELIC_DATA = {"GIRYA": (0, 3), "DU_VU_DOLL": (0, 3), "HAPPY_FLOWER": (0, 2), "INCENSE_BURNER": (0, 5),
              "PEN_NIB": (0, 9), "INK_BOTTLE": (0, 9), "NUNCHAKU": (0, 9), "SUNDIAL": (0, 2), "LIZARD_TAIL": (0, 1)}
STARTER = [("ASCENDERS_BANE", 0)] + [(c, 0) for c in ["STRIKE_RED"] * 4 + ["DEFEND_RED"] * 4 + ["BASH"]]


def deck_pool():
    from jaxsim._core.tables import cards as tc
    return [c for c in jaxsim.supported_cards() if c not in tc.NOT_IN_DECKS + ("ASCENDERS_BANE",)]


def deck_card(rng: random.Random, card: str, up: int):
    """A deck entry with the run data some cards carry (Ritual Dagger's damage, Searing Blow's upgrades)."""
    if card == "RITUAL_DAGGER":
        return (card, up, rng.randint(15, 40))
    return (card, rng.randint(1, 6) if card == "SEARING_BLOW" and up else up)


def random_deck(rng: random.Random):
    pool = deck_pool()
    deck = list(STARTER)
    for _ in range(rng.randint(0, 14)):
        deck.append(deck_card(rng, rng.choice(pool), rng.randint(0, 1)))
    if rng.random() < 0.3:
        deck.append(("WOUND", 0))
    return deck


def random_relics(rng):
    """0-5 relics: mostly ones with combat effects, sometimes a no-effect one; with run data where it matters."""
    from jaxsim._core.tables import relics as tr
    pool = sorted(tr.SUPPORTED) + sorted(tr.NO_EFFECT)[:5]
    names = rng.sample(pool, rng.randint(0, 5))
    return [relic_entry(rng, n) for n in names]


def relic_entry(rng, name):
    return (name, rng.randint(*RELIC_DATA[name])) if name in RELIC_DATA else (name, 0)


def random_potions(rng):
    """0-2 potions (A20 capacity 2) from the supported ones."""
    from jaxsim._core.tables import potions as tp
    return [rng.choice(tp.SUPPORTED_NAMES) for _ in range(rng.randint(0, 2))]


def fight_setup(rng, fight):
    """(deck, hp) for fight kind fight % 6: uniform random decks; strong decks at 300 hp; powers at 250 hp."""
    deck, hp = random_deck(rng), rng.randint(30, 80)
    if fight % 3 == 2:  # powers and the cards that trigger them, long fights
        deck += [(rng.choice(POWER_MIX), rng.randint(0, 1)) for _ in range(rng.randint(2, 5))]
        deck += [(rng.choice(HOOK_FUEL), rng.randint(0, 1)) for _ in range(rng.randint(2, 6))]
        return deck, 250
    if fight % 2:
        deck += [(c, 1) for c in ["INFLAME", "TWIN_STRIKE", "POMMEL_STRIKE", "SHRUG_IT_OFF"] * rng.randint(1, 3)]
        hp = 300  # long fights: reach phase 2, Execute and wins
    return deck, hp

