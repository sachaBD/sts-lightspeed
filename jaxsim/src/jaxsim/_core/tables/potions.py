"""Potion table: one row per potion id with effect columns (Sacred Bark doubles the amounts). See _core/potions.py.
Potions not listed are not supported yet (make_start rejects them); Fairy in a Bottle is passive."""
import numpy as np

from jaxsim._core.tables import ids

COLUMNS = ("supported", "drinkable", "block", "heal_pct", "dex", "lose_dex", "energy", "dmg_all", "dmg_target",
           "e_vuln", "e_weak", "str", "lose_str", "max_hp", "draw", "artifact", "plated", "metallicize", "thorns",
           "intangible", "regen", "upgrade_hand")
_COL = {c: i for i, c in enumerate(COLUMNS)}
ID = {n: i for i, n in enumerate(ids.POTIONS)}

_ROWS = {
    "BLOCK_POTION": dict(block=12), "BLOOD_POTION": dict(heal_pct=20), "DEXTERITY_POTION": dict(dex=2),
    "ENERGY_POTION": dict(energy=2), "EXPLOSIVE_POTION": dict(dmg_all=10), "FEAR_POTION": dict(e_vuln=3),
    "FIRE_POTION": dict(dmg_target=20), "FLEX_POTION": dict(str=5, lose_str=5), "FRUIT_JUICE": dict(max_hp=5),
    "STRENGTH_POTION": dict(str=2), "SWIFT_POTION": dict(draw=3), "WEAK_POTION": dict(e_weak=3),
    "ANCIENT_POTION": dict(artifact=1), "ESSENCE_OF_STEEL": dict(plated=4), "HEART_OF_IRON": dict(metallicize=6),
    "LIQUID_BRONZE": dict(thorns=3), "GHOST_IN_A_JAR": dict(intangible=1), "REGEN_POTION": dict(regen=5),
    "SPEED_POTION": dict(dex=5, lose_dex=5), "BLESSING_OF_THE_FORGE": dict(upgrade_hand=1),
    "FAIRY_POTION": dict(drinkable=0),
}


def _build():
    t = np.zeros((len(ids.POTIONS), len(COLUMNS)), np.int32)
    for name, row in _ROWS.items():
        for col, v in {"supported": 1, "drinkable": 1, **row}.items():
            t[ID[name], _COL[col]] = v
    return t


TABLE = _build()
SUPPORTED_NAMES = tuple(sorted(_ROWS))


def col(name: str) -> int:
    return _COL[name]
