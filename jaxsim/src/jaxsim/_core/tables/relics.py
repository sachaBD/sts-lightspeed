"""Relic support table. Relics are flags (State.rel, one per relic id) plus a few named counters.

NO_EFFECT: relics with no effect inside an A20 Champ fight that changes a win/loss result (shop, gold, map, rest,
card reward relics; post-combat heals; relics applied when picked up; Preserved Insect and Gremlin Horn need
elites / several enemies). Accepted and ignored.
SUPPORTED: relics with in-combat effects that jaxsim implements (see _core/relics.py).
"""
from jaxsim._core.tables import ids

NO_EFFECT = {
    "BURNING_BLOOD", "BLACK_BLOOD", "MEAT_ON_THE_BONE", "OMAMORI", "JUZU_BRACELET", "MAW_BANK", "MEAL_TICKET",
    "REGAL_PILLOW", "SMILING_MASK", "STRAWBERRY", "MANGO", "PEAR", "OLD_COIN", "TINY_CHEST", "QUESTION_CARD",
    "SINGING_BOWL", "THE_COURIER", "PRAYER_WHEEL", "WING_BOOTS", "SHOVEL", "PEACE_PIPE", "ETERNAL_FEATHER",
    "DARKSTONE_PERIAPT", "MATRYOSHKA", "MOLTEN_EGG", "TOXIC_EGG", "FROZEN_EGG", "WHETSTONE", "WAR_PAINT",
    "POTION_BELT", "CERAMIC_FISH", "GOLDEN_IDOL", "PRESERVED_INSECT", "PRISMATIC_SHARD", "BLACK_STAR",
    "EMPTY_CAGE", "ORRERY", "PANDORAS_BOX", "CALLING_BELL", "ASTROLABE", "TINY_HOUSE", "LEES_WAFFLE",
    "NLOTHS_GIFT", "NLOTHS_HUNGRY_FACE", "MEMBERSHIP_CARD", "DREAM_CATCHER", "SSSERPENT_HEAD", "SPIRIT_POOP",
    "WHITE_BEAST_STATUE", "GREMLIN_HORN", "SLING_OF_COURAGE", "THE_SPECIMEN", "BLOODY_IDOL", "CIRCLET", "RED_CIRCLET", "NEOWS_LAMENT",
    "BOTTLED_FLAME", "BOTTLED_LIGHTNING", "BOTTLED_TORNADO",   # bottled cards themselves are out of scope
}

SUPPORTED = {
    # combat start: stats and energy
    "VAJRA", "ODDLY_SMOOTH_STONE", "GIRYA", "DU_VU_DOLL", "MUTAGENIC_STRENGTH", "BRIMSTONE", "PHILOSOPHERS_STONE",
    "ECTOPLASM", "SOZU", "VELVET_CHOKER", "BUSTED_CROWN", "COFFEE_DRIPPER", "CURSED_KEY", "FUSION_HAMMER",
    "SLAVERS_COLLAR", "MARK_OF_PAIN",
    # combat start: other
    "ANCHOR", "BLOOD_VIAL", "PANTOGRAPH", "LANTERN", "BAG_OF_MARBLES", "BAG_OF_PREPARATION", "RED_MASK",
    "BRONZE_SCALES", "THREAD_AND_NEEDLE", "FOSSILIZED_HELIX", "CLOCKWORK_SOUVENIR", "GREMLIN_VISAGE",
    # turn start
    "MERCURY_HOURGLASS", "HORN_CLEAT", "CAPTAINS_WHEEL", "HAPPY_FLOWER", "ART_OF_WAR", "INCENSE_BURNER",
    # batch 2: card play counters
    "PEN_NIB", "KUNAI", "SHURIKEN", "ORNAMENTAL_FAN", "NUNCHAKU", "INK_BOTTLE", "LETTER_OPENER", "BIRD_FACED_URN",
    # batch 2: damage, defence, hp loss, healing
    "STRIKE_DUMMY", "PAPER_PHROG", "THE_BOOT", "CHAMPION_BELT", "TORII", "TUNGSTEN_ROD", "GINGER", "TURNIP",
    "SELF_FORMING_CLAY", "RED_SKULL", "LIZARD_TAIL", "MAGIC_FLOWER",
    "MARK_OF_THE_BLOOM",
    # batch 2: turn / pile triggers
    "ORICHALCUM", "CALIPERS", "ICE_CREAM", "POCKETWATCH", "CHARONS_ASHES", "SUNDIAL", "UNCEASING_TOP",
    # potion relics
    "SACRED_BARK", "TOY_ORNITHOPTER",
}

ID = {n: i for i, n in enumerate(ids.RELICS)}
ENERGY = ("ECTOPLASM", "SOZU", "VELVET_CHOKER", "BUSTED_CROWN", "COFFEE_DRIPPER", "CURSED_KEY", "FUSION_HAMMER",
          "SLAVERS_COLLAR", "MARK_OF_PAIN", "PHILOSOPHERS_STONE")
