"""Scenario parity (gate): small fixed fights per mechanism, compared exactly with the C++ step by step.

Groups, one test each (a failing test lists every failing scenario; SCENARIO=substring runs only matching ones):
- cards: one per card (two copies, unupgraded and upgraded, in a 5-card deck, so the hand is the whole deck),
  played first whenever legal for 3 turns; and again against a 12-hp Champ (killing-blow paths);
- relics and potions: one each, in a small deck;
- combos: hand-built decks of interacting mechanisms (powers with their triggers, choices, curses), played by a
  seeded random policy;
- rules: hand-set piles and a scripted line of play for the edge cases random play rarely reaches.

A scenario that fails on a known, accepted difference goes in KNOWN_DIFFERENCES with the reason. Everything
else must match exactly.
"""
import os
import random

import pytest

import jaxsim
from tests.oracle import decks
from tests.oracle.harness import SPEC, Fight

CARD = {n: i for i, n in enumerate(jaxsim.CARD_NAMES)}
FILLER = [("STRIKE_RED", 0), ("DEFEND_RED", 0), ("BASH", 0)]
SMALL = [("BASH", 0), ("STRIKE_RED", 0), ("DEFEND_RED", 0), ("INFLAME", 0), ("WOUND", 0)]

# scenario name -> why the difference is accepted (skipped in the group's verdict)
KNOWN_DIFFERENCES: dict[str, str] = {}

COMBOS = {
    "exhaust_engine": ["CORRUPTION", "FEEL_NO_PAIN", "DARK_EMBRACE", "SECOND_WIND", "TRUE_GRIT", "SHRUG_IT_OFF",
                       "BURNING_PACT", "FIEND_FIRE", "SEVER_SOUL", "DEFEND_RED", "DEFEND_RED", "WOUND"],
    "statuses": ["EVOLVE", "FIRE_BREATHING", "WILD_STRIKE", "POWER_THROUGH", "IMMOLATE", "RECKLESS_CHARGE", "WOUND",
                 "WOUND", "STRIKE_RED", "STRIKE_RED", "DEFEND_RED"],
    "strength": ["INFLAME", "LIMIT_BREAK", "DEMON_FORM", "HEAVY_BLADE", "TWIN_STRIKE", "SWORD_BOOMERANG", "FLEX",
                 "WHIRLWIND", "STRIKE_RED", "STRIKE_RED", "DEFEND_RED"],
    "self_damage": ["RUPTURE", "OFFERING", "HEMOKINESIS", "BLOODLETTING", "BLOOD_FOR_BLOOD", "COMBUST", "BRUTALITY",
                    "REAPER", "STRIKE_RED", "DEFEND_RED"],
    "block": ["BARRICADE", "METALLICIZE", "JUGGERNAUT", "ENTRENCH", "BODY_SLAM", "FLAME_BARRIER", "IMPERVIOUS",
              "GHOSTLY_ARMOR", "DEFEND_RED", "DEFEND_RED", "SHRUG_IT_OFF"],
    "rampage": ["RAMPAGE", "RAMPAGE", "ANGER", "PERFECTED_STRIKE", "STRIKE_RED", "STRIKE_RED", "HEADBUTT",
                "EXHUME", "ARMAMENTS", "CLASH"],
    "choices": ["WARCRY", "DUAL_WIELD", "HEADBUTT", "ARMAMENTS", "EXHUME", "TRUE_GRIT", "BURNING_PACT",
                "SECOND_WIND", "STRIKE_RED", "DEFEND_RED", "BASH"],
    "curses": ["NORMALITY", "PAIN", "REGRET", "DECAY", "DOUBT", "SHAME", "NECRONOMICURSE", "STRIKE_RED",
               "STRIKE_RED", "DEFEND_RED", "POMMEL_STRIKE", "SHRUG_IT_OFF"],
}


def _is_curse(name):
    from jaxsim._core.tables import cards as tc
    return int(tc.TABLE[CARD[name], 0][tc.col("type")]) == tc.CURSE


def scenarios():
    from jaxsim._core.tables import potions as tp, relics as tr
    out = []
    for c in decks.deck_pool():
        rng = random.Random(c)
        ups = (0, 0) if _is_curse(c) else (0, 1)
        deck = [decks.deck_card(rng, c, u) for u in ups] + FILLER
        out.append(dict(name=f"card:{c}", deck=deck, hp=80, focus={c}, turns=3))
        out.append(dict(name=f"card_kill:{c}", deck=deck, hp=80, focus={c}, turns=2, enemy_hp=12))
    for r in sorted(tr.SUPPORTED):
        data = decks.RELIC_DATA.get(r, (0, 0))[1]
        out.append(dict(name=f"relic:{r}", deck=SMALL, hp=60, relics=[(r, data)], turns=3))
    for p in tp.SUPPORTED_NAMES:
        out.append(dict(name=f"potion:{p}", deck=FILLER + [("STRIKE_RED", 0), ("DEFEND_RED", 0)], hp=60,
                        potions=[p], turns=2))
    for name, deck in COMBOS.items():
        for seed in range(3):
            out.append(dict(name=f"combo:{name}:{seed}", deck=[(c, seed % 2) for c in deck], hp=250, turns=5,
                            seed=seed))
    return out


SCENARIOS = scenarios()


def pick(f: Fight, focus: set[str], rng: random.Random | None) -> int:
    """Deterministic: potions first, then the focus cards, then the lowest slot; END_TURN when nothing else.
    With rng (combos): a seeded random non-END_TURN action."""
    legal = sorted(f.legal())
    other = [a for a in legal if a != SPEC.END_TURN]
    if not other:
        return SPEC.END_TURN
    if rng is not None:
        return rng.choice(other)
    potions = [a for a in other if a > SPEC.END_TURN]
    ids = {CARD[c] for c in focus}
    focused = [a for a in other if a < SPEC.D and int(f.obs.slot_card[a]) in ids]
    return (potions or focused or other)[0]


RULES = [
    # name, piles and fields set after the opening draw, scripted actions: "end", ("play"|"pick", card[, up]),
    # ("potion", name). The deck is every pile card unless given.
    dict(name="iron_wave_dex_then_frail", hp=80, potions=["DEXTERITY_POTION"],
         setup=dict(hand=["IRON_WAVE", "IRON_WAVE", "DEFEND_RED"], draw=["IRON_WAVE"] * 3 + ["DEFEND_RED"] * 3,
                    enemy_move="THE_CHAMP_FACE_SLAP", energy=3),
         actions=[("play", "IRON_WAVE"), "end", ("play", "IRON_WAVE"), ("potion", "DEXTERITY_POTION"),
                  ("play", "IRON_WAVE")]),
    dict(name="feed_kill_raises_max_hp", hp=50, setup=dict(hand=["FEED"], enemy_hp=5), actions=[("play", "FEED")]),
    dict(name="reaper_kill_heals", hp=50, setup=dict(hand=["REAPER"], enemy_hp=3), actions=[("play", "REAPER")]),
    dict(name="reaper_heals", hp=50, setup=dict(hand=["REAPER"]), actions=[("play", "REAPER")]),
    dict(name="clash_with_attacks", setup=dict(hand=["CLASH", "STRIKE_RED"]), actions=[("play", "CLASH")]),
    dict(name="clash_blocked_by_skill", setup=dict(hand=["CLASH", "DEFEND_RED"]), actions=["end"]),
    dict(name="burn_through_block", setup=dict(hand=["BURN", "BURN"], block=3, enemy_move="THE_CHAMP_GLOAT"),
         actions=["end"]),
    dict(name="power_through_fills_hand", setup=dict(hand=["POWER_THROUGH"] + ["STRIKE_RED"] * 8),
         actions=[("play", "POWER_THROUGH")]),
    dict(name="power_through_overflow_to_discard", setup=dict(hand=["POWER_THROUGH"] + ["STRIKE_RED"] * 9),
         actions=[("play", "POWER_THROUGH")]),
    dict(name="battle_trance_no_draw", setup=dict(hand=["BATTLE_TRANCE", "POMMEL_STRIKE"], draw=["STRIKE_RED"] * 10),
         actions=[("play", "BATTLE_TRANCE"), ("play", "POMMEL_STRIKE")]),
    dict(name="heavy_blade_strength", setup=dict(hand=[("INFLAME", 1), "HEAVY_BLADE"]),
         actions=[("play", "INFLAME"), ("play", "HEAVY_BLADE")]),
    dict(name="perfected_strike_counts_outside_exhaust",
         setup=dict(hand=["PERFECTED_STRIKE", "STRIKE_RED", "TWIN_STRIKE", "DEFEND_RED"], discard=["STRIKE_RED"] * 2,
                    exhaust=["TWIN_STRIKE"] * 5),
         actions=[("play", "PERFECTED_STRIKE")]),
    dict(name="whirlwind_x", setup=dict(hand=["WHIRLWIND", "WHIRLWIND"], draw=["DEFEND_RED"] * 5),
         actions=[("play", "WHIRLWIND"), ("play", "WHIRLWIND")]),
    dict(name="rampage_grows_per_copy", setup=dict(hand=["RAMPAGE", "RAMPAGE"], draw=["STRIKE_RED"] * 5),
         actions=[("play", "RAMPAGE"), ("play", "RAMPAGE"), "end"]),
    dict(name="blood_for_blood_per_hit", setup=dict(hand=["BLOOD_FOR_BLOOD", "STRIKE_RED"], block=5,
                                                    draw=["DEFEND_RED"] * 5, enemy_move="THE_CHAMP_EXECUTE"),
         actions=["end"]),
    dict(name="second_wind_exhausts_non_attacks",
         setup=dict(hand=["SECOND_WIND", "SENTINEL", "DEFEND_RED", "STRIKE_RED"]), actions=[("play", "SECOND_WIND")]),
    dict(name="sever_soul_kill_keeps_hand", setup=dict(hand=["SEVER_SOUL", "DEFEND_RED"], enemy_hp=5),
         actions=[("play", "SEVER_SOUL")]),
    dict(name="flame_barrier_kills_on_champ_turn",
         setup=dict(hand=["FLAME_BARRIER"], block=50, enemy_hp=6, enemy_move="THE_CHAMP_EXECUTE"),
         actions=[("play", "FLAME_BARRIER"), "end"]),
    dict(name="juggernaut_per_block_gain", setup=dict(hand=["JUGGERNAUT", "DEFEND_RED"]),
         actions=[("play", "JUGGERNAUT"), ("play", "DEFEND_RED")]),
    dict(name="rupture_after_hemokinesis", setup=dict(hand=["RUPTURE", "HEMOKINESIS"]),
         actions=[("play", "RUPTURE"), ("play", "HEMOKINESIS")]),
    dict(name="corruption_feel_no_pain_dark_embrace",
         setup=dict(hand=["FEEL_NO_PAIN", "DARK_EMBRACE", "CORRUPTION", "SHRUG_IT_OFF", "DEFEND_RED"],
                    draw=["SHRUG_IT_OFF", "STRIKE_RED", "STRIKE_RED"], energy=10),
         actions=[("play", "FEEL_NO_PAIN"), ("play", "DARK_EMBRACE"), ("play", "CORRUPTION"), ("play", "SHRUG_IT_OFF"),
                  ("play", "DEFEND_RED"), ("play", "SHRUG_IT_OFF")]),
    dict(name="barricade_keeps_block_berserk_energy",
         setup=dict(hand=["BARRICADE", "BERSERK", "DEFEND_RED"], block=30, energy=10, draw=["STRIKE_RED"] * 5,
                    enemy_move="THE_CHAMP_GLOAT"),
         actions=[("play", "BARRICADE"), ("play", "BERSERK"), "end"]),
    dict(name="combust_and_metallicize_end_of_turn",
         setup=dict(hand=["COMBUST", "COMBUST", "METALLICIZE"], energy=10, draw=["STRIKE_RED"] * 5,
                    enemy_move="THE_CHAMP_GLOAT"),
         actions=[("play", "COMBUST"), ("play", "COMBUST"), ("play", "METALLICIZE"), "end"]),
    dict(name="headbutt_pick_goes_on_top", setup=dict(hand=["HEADBUTT", "STRIKE_RED", "DEFEND_RED"],
                                                      discard=["STRIKE_RED", "DEFEND_RED"], draw=["BASH"] * 5),
         actions=[("play", "HEADBUTT"), ("pick", "DEFEND_RED"), "end"]),
    dict(name="armaments_single_option_resolves", setup=dict(hand=["ARMAMENTS", "STRIKE_RED"]),
         actions=[("play", "ARMAMENTS")]),
    dict(name="dual_wield_copies", setup=dict(hand=["DUAL_WIELD", "BASH"]), actions=[("play", "DUAL_WIELD")]),
    dict(name="exhume_returns_a_card", setup=dict(hand=["EXHUME", "STRIKE_RED"], exhaust=["STRIKE_RED", "BASH"]),
         actions=[("play", "EXHUME"), ("pick", "BASH")]),
    dict(name="velvet_choker_six_plays", relics=[("VELVET_CHOKER", 0)],
         setup=dict(hand=["ANGER"] * 6 + ["STRIKE_RED"], energy=10, draw=["STRIKE_RED"] * 5),
         actions=[("play", "ANGER")] * 6),   # then nothing is legal but END_TURN (checked against the C++)
    dict(name="fire_potions", potions=["FIRE_POTION", "FIRE_POTION"], deck=["STRIKE_RED"] * 10,
         actions=[("potion", "FIRE_POTION"), ("potion", "FIRE_POTION")]),
    dict(name="fairy_revives", hp=1, max_hp=80, potions=["FAIRY_POTION"], deck=["STRIKE_RED"] * 10,
         setup=dict(enemy_move="THE_CHAMP_HEAVY_SLASH"), actions=["end"]),
    dict(name="relics_at_combat_start", hp=50, max_hp=80, deck=["STRIKE_RED"] * 10,
         relics=[("ANCHOR", 0), ("VAJRA", 0), ("LANTERN", 0), ("BLOOD_VIAL", 0), ("GIRYA", 2), ("SOZU", 0)],
         actions=["end"]),
] + [dict(name=f"brutality_plus_innate:{seed}", seed=seed, deck=[("BRUTALITY", 1)] + ["STRIKE_RED"] * 20, actions=[])
     for seed in range(3)]


def _entry(c):
    return (c, 0) if isinstance(c, str) else c


def run_rule(oracle, sc):
    setup = {k: ([_entry(c) for c in v] if k in ("hand", "draw", "discard", "exhaust") else v)
             for k, v in sc.get("setup", {}).items()}
    deck = [_entry(c) for c in sc["deck"]] if "deck" in sc else [
        c for k in ("hand", "draw", "discard", "exhaust") for c in setup.get(k, [])]
    hp = sc.get("hp", 80)
    f = Fight(oracle, sc["name"], deck, hp, max_hp=sc.get("max_hp"), relics=sc.get("relics", ()),
              potions=sc.get("potions", ()), seed=sc.get("seed", 0), setup=setup)
    for act in sc["actions"]:
        assert not f.done, f"{sc['name']}: fight over before {act}"
        if act == "end":
            f.act(SPEC.END_TURN)
        elif act[0] == "potion":
            f.act(f.potion(act[1]))
        else:
            f.act(f.find(*act[1:]))


def run_generated(oracle, sc):
    f = Fight(oracle, sc["name"], sc["deck"], sc["hp"], max_hp=80 if sc["hp"] <= 80 else sc["hp"],
              relics=sc.get("relics", ()), potions=sc.get("potions", ()), seed=sc.get("seed", 0),
              enemy_hp=sc.get("enemy_hp"))
    rng = random.Random(sc["seed"]) if "seed" in sc else None
    last_turn = int(f.obs.turn) + sc["turns"]
    for _ in range(200):
        if f.done or int(f.obs.turn) >= last_turn:
            break
        f.act(pick(f, sc.get("focus", set()), rng))


def run_group(oracle, scenarios, runner):
    only = os.environ.get("SCENARIO")
    failures, ran = [], 0
    for sc in scenarios:
        if only and only not in sc["name"]:
            continue
        ran += 1
        try:
            runner(oracle, sc)
        except Exception as e:  # noqa: BLE001 - report every failing scenario, not just the first
            if sc["name"] not in KNOWN_DIFFERENCES:
                failures.append(f"{sc['name']}: {type(e).__name__}: {e}")
    if only and not ran:
        pytest.skip(f"no scenario matches SCENARIO={only}")
    assert not failures, f"{len(failures)}/{ran} scenarios differ: " + ", ".join(
        f.split(":")[0] + ":" + f.split(":")[1] for f in failures) + "\n\n" + "\n\n".join(failures[:3])


def _group(prefix):
    return [s for s in SCENARIOS if s["name"].startswith(prefix)]


def test_card_scenarios(oracle):
    run_group(oracle, _group("card"), run_generated)


def test_relic_scenarios(oracle):
    run_group(oracle, _group("relic:"), run_generated)


def test_potion_scenarios(oracle):
    run_group(oracle, _group("potion:"), run_generated)


def test_combo_scenarios(oracle):
    run_group(oracle, _group("combo:"), run_generated)


def test_rule_scenarios(oracle):
    run_group(oracle, RULES, run_rule)
