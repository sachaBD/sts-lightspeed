"""Card table: one row per (card id, upgraded) with fixed effect columns, applied by cards.play_pre / play_post.

Effect order inside a play (matches the C++ action order for every supported card):
    self hp loss -> block -> damage x hits -> specials on damage (Reaper, Feed, Dropkick check) ->
    enemy debuffs -> player buffs -> energy -> draw -> created cards / Battle Trance -> the played card leaves.
After a winning hit only block, hp loss/heal and the card leaving still happen (C++ clearOnCombatVictory=false).

cost / type / exhausts / ethereal are checked against the C++ card data by tests/test_parity.py.
Only cards listed in _ROWS are supported; everything else has supported=0 (make_start rejects it).
"""
import numpy as np

from jaxsim._core.tables import ids

ATTACK, SKILL, POWER, STATUS, CURSE = range(5)

# create_pile values
TO_DRAW, TO_DISCARD, TO_HAND = 1, 2, 3

# special values: small dedicated rules, each a few masked scalar ops in cards.py
(SP_NONE, SP_CLASH, SP_DROPKICK, SP_SPOT_WEAKNESS, SP_ENTRENCH, SP_LIMIT_BREAK, SP_FEED, SP_REAPER,
 SP_BATTLE_TRANCE) = range(9)

COLUMNS = ("supported", "cost", "type", "playable", "ethereal", "exhausts",
           "hp_loss", "block", "damage", "hits", "e_vuln", "e_weak", "e_str", "p_str", "lose_str", "energy", "draw",
           "copy_to_discard", "create_card", "create_pile", "create_n", "eot_damage", "special", "special_amt")
_COL = {c: i for i, c in enumerate(COLUMNS)}

_ID = {n: i for i, n in enumerate(ids.CARDS)}
W, DAZED, BURN = _ID["WOUND"], _ID["DAZED"], _ID["BURN"]

# name: (base row, upgrade overrides). Unlisted columns are 0; hits defaults to 1 for attacks with damage.
_ROWS = {
    # starter + PR #2 set
    "STRIKE_RED":      (dict(cost=1, type=ATTACK, damage=6), dict(damage=9)),
    "DEFEND_RED":      (dict(cost=1, type=SKILL, block=5), dict(block=8)),
    "BASH":            (dict(cost=2, type=ATTACK, damage=8, e_vuln=2), dict(damage=10, e_vuln=3)),
    "POMMEL_STRIKE":   (dict(cost=1, type=ATTACK, damage=9, draw=1), dict(damage=10, draw=2)),
    "SHRUG_IT_OFF":    (dict(cost=1, type=SKILL, block=8, draw=1), dict(block=11)),
    "INFLAME":         (dict(cost=1, type=POWER, p_str=2), dict(p_str=3)),
    "TWIN_STRIKE":     (dict(cost=1, type=ATTACK, damage=5, hits=2), dict(damage=7)),
    "IRON_WAVE":       (dict(cost=1, type=ATTACK, damage=5, block=5), dict(damage=7, block=7)),
    "ANGER":           (dict(cost=0, type=ATTACK, damage=6, copy_to_discard=1), dict(damage=8)),
    "WILD_STRIKE":     (dict(cost=1, type=ATTACK, damage=12, create_card=W, create_pile=TO_DRAW, create_n=1),
                        dict(damage=17)),
    # batch 1: table rows and small columns
    "CLEAVE":          (dict(cost=1, type=ATTACK, damage=8), dict(damage=11)),
    "CLOTHESLINE":     (dict(cost=2, type=ATTACK, damage=12, e_weak=2), dict(damage=14, e_weak=3)),
    "THUNDERCLAP":     (dict(cost=1, type=ATTACK, damage=4, e_vuln=1), dict(damage=7)),
    "UPPERCUT":        (dict(cost=2, type=ATTACK, damage=13, e_weak=1, e_vuln=1), dict(e_weak=2, e_vuln=2)),
    "SHOCKWAVE":       (dict(cost=2, type=SKILL, e_weak=3, e_vuln=3, exhausts=1), dict(e_weak=5, e_vuln=5)),
    "INTIMIDATE":      (dict(cost=0, type=SKILL, e_weak=1, exhausts=1), dict(e_weak=2)),
    "DISARM":          (dict(cost=1, type=SKILL, e_str=-2, exhausts=1), dict(e_str=-3)),
    "SWORD_BOOMERANG": (dict(cost=1, type=ATTACK, damage=3, hits=3), dict(hits=4)),
    "PUMMEL":          (dict(cost=1, type=ATTACK, damage=2, hits=4, exhausts=1), dict(hits=5)),
    "BLUDGEON":        (dict(cost=3, type=ATTACK, damage=32), dict(damage=42)),
    "CARNAGE":         (dict(cost=2, type=ATTACK, damage=20, ethereal=1), dict(damage=28)),
    "RECKLESS_CHARGE": (dict(cost=0, type=ATTACK, damage=7, create_card=DAZED, create_pile=TO_DRAW, create_n=1),
                        dict(damage=10)),
    "POWER_THROUGH":   (dict(cost=1, type=SKILL, block=15, create_card=W, create_pile=TO_HAND, create_n=2),
                        dict(block=20)),
    "IMMOLATE":        (dict(cost=2, type=ATTACK, damage=21, create_card=BURN, create_pile=TO_DISCARD, create_n=1),
                        dict(damage=28)),
    "SEEING_RED":      (dict(cost=1, type=SKILL, energy=2, exhausts=1), dict(cost=0)),
    "GHOSTLY_ARMOR":   (dict(cost=1, type=SKILL, block=10, ethereal=1), dict(block=13)),
    "BATTLE_TRANCE":   (dict(cost=0, type=SKILL, draw=3, special=SP_BATTLE_TRANCE), dict(draw=4)),
    "OFFERING":        (dict(cost=0, type=SKILL, hp_loss=6, energy=2, draw=3, exhausts=1), dict(draw=5)),
    "IMPERVIOUS":      (dict(cost=2, type=SKILL, block=30, exhausts=1), dict(block=40)),
    "BLOODLETTING":    (dict(cost=0, type=SKILL, hp_loss=3, energy=2), dict(energy=3)),
    "HEMOKINESIS":     (dict(cost=1, type=ATTACK, hp_loss=2, damage=15), dict(damage=20)),
    "FLEX":            (dict(cost=0, type=SKILL, p_str=2, lose_str=2), dict(p_str=4, lose_str=4)),
    "CLASH":           (dict(cost=0, type=ATTACK, damage=14, special=SP_CLASH), dict(damage=18)),
    "DROPKICK":        (dict(cost=1, type=ATTACK, damage=5, special=SP_DROPKICK), dict(damage=8)),
    "SPOT_WEAKNESS":   (dict(cost=1, type=SKILL, special=SP_SPOT_WEAKNESS, special_amt=3), dict(special_amt=4)),
    "ENTRENCH":        (dict(cost=2, type=SKILL, special=SP_ENTRENCH), dict(cost=1)),
    "LIMIT_BREAK":     (dict(cost=1, type=SKILL, special=SP_LIMIT_BREAK, exhausts=1), dict(exhausts=0)),
    "SENTINEL":        (dict(cost=1, type=SKILL, block=5), dict(block=8)),   # on-exhaust energy: with exhaust sources
    "FEED":            (dict(cost=1, type=ATTACK, damage=10, special=SP_FEED, special_amt=3, exhausts=1),
                        dict(damage=12, special_amt=4)),
    "REAPER":          (dict(cost=2, type=ATTACK, damage=4, special=SP_REAPER, exhausts=1), dict(damage=5)),
    # statuses and curses
    "WOUND":           (dict(cost=-2, type=STATUS, playable=0), {}),
    "DAZED":           (dict(cost=-2, type=STATUS, playable=0, ethereal=1), {}),
    "BURN":            (dict(cost=-2, type=STATUS, playable=0, eot_damage=2), dict(eot_damage=4)),
    "ASCENDERS_BANE":  (dict(cost=-3, type=CURSE, playable=0, ethereal=1), {}),   # C++ cost -3
}
NOT_IN_DECKS = ("WOUND", "DAZED", "BURN")   # created in combat only (still valid in a deck)


def _build() -> np.ndarray:
    t = np.zeros((len(ids.CARDS), 2, len(COLUMNS)), np.int32)
    for name, (base, up) in _ROWS.items():
        row = {"supported": 1, "playable": 1, "hits": 1 if base.get("damage") else 0, **base}
        for u, over in ((0, {}), (1, up)):
            for col, v in {**row, **over}.items():
                t[_ID[name], u, _COL[col]] = v
    return t


TABLE = _build()                      # [n_cards, 2, len(COLUMNS)]
SUPPORTED_NAMES = tuple(sorted(_ROWS))


def col(name: str) -> int:
    return _COL[name]
