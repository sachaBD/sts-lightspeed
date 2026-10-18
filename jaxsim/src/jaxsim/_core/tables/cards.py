"""Card table: one row per (card id, upgraded) with fixed effect columns, applied by cards.resolve.

Effect order inside resolve (matches the C++ action order for every supported card):
    block -> damage x hits -> enemy debuffs -> player buffs -> draw -> created cards -> played card leaves.
Only cards listed in _ROWS are supported; everything else has supported=0 (make_start rejects it).
"""
import numpy as np

from jaxsim._core.tables import ids

ATTACK, SKILL, POWER, STATUS, CURSE = range(5)

COLUMNS = ("supported", "cost", "type", "playable", "ethereal", "exhausts", "damage", "hits", "block",
           "e_vuln", "e_weak", "p_str", "draw", "copy_to_discard", "add_draw_card")
_COL = {c: i for i, c in enumerate(COLUMNS)}

_ID = {n: i for i, n in enumerate(ids.CARDS)}

# name: (base row, upgrade overrides). Unlisted columns are 0; hits defaults to 1 for attacks.
_ROWS = {
    "STRIKE_RED":     (dict(cost=1, type=ATTACK, damage=6), dict(damage=9)),
    "DEFEND_RED":     (dict(cost=1, type=SKILL, block=5), dict(block=8)),
    "BASH":           (dict(cost=2, type=ATTACK, damage=8, e_vuln=2), dict(damage=10, e_vuln=3)),
    "POMMEL_STRIKE":  (dict(cost=1, type=ATTACK, damage=9, draw=1), dict(damage=10, draw=2)),
    "SHRUG_IT_OFF":   (dict(cost=1, type=SKILL, block=8, draw=1), dict(block=11)),
    "INFLAME":        (dict(cost=1, type=POWER, p_str=2), dict(p_str=3)),
    "TWIN_STRIKE":    (dict(cost=1, type=ATTACK, damage=5, hits=2), dict(damage=7)),
    "IRON_WAVE":      (dict(cost=1, type=ATTACK, damage=5, block=5), dict(damage=7, block=7)),
    "ANGER":          (dict(cost=0, type=ATTACK, damage=6, copy_to_discard=1), dict(damage=8)),
    "WILD_STRIKE":    (dict(cost=1, type=ATTACK, damage=12, add_draw_card=_ID["WOUND"]), dict(damage=17)),
    "WOUND":          (dict(cost=-1, type=STATUS, playable=0), {}),
    "ASCENDERS_BANE": (dict(cost=-1, type=CURSE, playable=0, ethereal=1), {}),
}


def _build() -> np.ndarray:
    t = np.zeros((len(ids.CARDS), 2, len(COLUMNS)), np.int32)
    for name, (base, up) in _ROWS.items():
        row = {"supported": 1, "playable": 1, "hits": 1 if base.get("type") == ATTACK else 0, **base}
        for u, over in ((0, {}), (1, up)):
            r = {**row, **over}
            for col, v in r.items():
                t[_ID[name], u, _COL[col]] = v
    return t


TABLE = _build()                      # [n_cards, 2, len(COLUMNS)]
SUPPORTED_NAMES = tuple(sorted(_ROWS))


def col(name: str) -> int:
    return _COL[name]
