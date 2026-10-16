"""Static data tables for the JAX Champ-fight slice.

Every card is a row of data, not a branch of code: cost, type, flags and up to
MAX_EFFECTS micro-ops (opcode, a, b) that are pushed onto the action queue when
the card is played. Upgraded and unupgraded versions are separate rows
(row = card_id * 2 + upgraded), so all numbers are static.
"""
import numpy as np

# ---------------------------------------------------------------- micro-ops
# The VM in engine.py dispatches on these. Keep the count small: under vmap
# every opcode body runs for every lane on every VM iteration.
OP_NOP = 0
OP_ATTACK = 1           # a = base damage, b = hits
OP_BLOCK = 2            # a = base block
OP_ENEMY_STATUS = 3     # a = enemy status idx, b = amount
OP_PLAYER_STATUS = 4    # a = player status idx, b = amount
OP_DRAW = 5             # a = count (re-pushes itself with a-1)
OP_ENERGY = 6           # a = amount
OP_ADD_DISCARD = 7      # a = card id, b = upgraded
OP_ADD_DRAW_RANDOM = 8  # a = card id, b = upgraded
OP_START_TURN = 9
OP_END_TURN = 10
OP_ENEMY_TURN = 11
OP_POST_PLAY = 12       # a = card slot just played
NUM_OPS = 13

# ---------------------------------------------------------------- statuses
P_STR, P_DEX, P_VULN, P_WEAK, P_FRAIL = range(5)
NUM_PLAYER_STATUS = 8
E_STR, E_VULN, E_WEAK, E_METALLICIZE = range(4)
NUM_ENEMY_STATUS = 4

# ---------------------------------------------------------------- cards
ATTACK, SKILL, POWER, STATUS, CURSE = range(5)

(EMPTY, STRIKE, DEFEND, BASH, POMMEL_STRIKE, SHRUG_IT_OFF, INFLAME, TWIN_STRIKE,
 IRON_WAVE, ANGER, WILD_STRIKE, WOUND) = range(12)
NUM_CARDS = 12
MAX_EFFECTS = 3

CARD_NAMES = ["EMPTY", "STRIKE", "DEFEND", "BASH", "POMMEL_STRIKE", "SHRUG_IT_OFF", "INFLAME",
              "TWIN_STRIKE", "IRON_WAVE", "ANGER", "WILD_STRIKE", "WOUND"]

# card: (cost, type, playable, exhaust, [effects unupgraded], [effects upgraded])
_CARDS = {
    STRIKE:        (1, ATTACK, 1, 0, [(OP_ATTACK, 6, 1)], [(OP_ATTACK, 9, 1)]),
    DEFEND:        (1, SKILL, 1, 0, [(OP_BLOCK, 5, 0)], [(OP_BLOCK, 8, 0)]),
    BASH:          (2, ATTACK, 1, 0, [(OP_ATTACK, 8, 1), (OP_ENEMY_STATUS, E_VULN, 2)],
                                     [(OP_ATTACK, 10, 1), (OP_ENEMY_STATUS, E_VULN, 3)]),
    POMMEL_STRIKE: (1, ATTACK, 1, 0, [(OP_ATTACK, 9, 1), (OP_DRAW, 1, 0)],
                                     [(OP_ATTACK, 10, 1), (OP_DRAW, 2, 0)]),
    SHRUG_IT_OFF:  (1, SKILL, 1, 0, [(OP_BLOCK, 8, 0), (OP_DRAW, 1, 0)],
                                    [(OP_BLOCK, 11, 0), (OP_DRAW, 1, 0)]),
    INFLAME:       (1, POWER, 1, 0, [(OP_PLAYER_STATUS, P_STR, 2)], [(OP_PLAYER_STATUS, P_STR, 3)]),
    TWIN_STRIKE:   (1, ATTACK, 1, 0, [(OP_ATTACK, 5, 2)], [(OP_ATTACK, 7, 2)]),
    IRON_WAVE:     (1, ATTACK, 1, 0, [(OP_BLOCK, 5, 0), (OP_ATTACK, 5, 1)],
                                     [(OP_BLOCK, 7, 0), (OP_ATTACK, 7, 1)]),
    ANGER:         (0, ATTACK, 1, 0, [(OP_ATTACK, 6, 1), (OP_ADD_DISCARD, ANGER, 0)],
                                     [(OP_ATTACK, 8, 1), (OP_ADD_DISCARD, ANGER, 1)]),
    WILD_STRIKE:   (1, ATTACK, 1, 0, [(OP_ATTACK, 12, 1), (OP_ADD_DRAW_RANDOM, WOUND, 0)],
                                     [(OP_ATTACK, 17, 1), (OP_ADD_DRAW_RANDOM, WOUND, 0)]),
    WOUND:         (0, STATUS, 0, 0, [], []),
}


def _build():
    rows = NUM_CARDS * 2
    cost = np.zeros(rows, np.int32)
    ctype = np.zeros(rows, np.int32)
    playable = np.zeros(rows, np.int32)
    exhaust = np.zeros(rows, np.int32)
    effects = np.zeros((rows, MAX_EFFECTS, 3), np.int32)
    for cid, (c, t, p, e, eff0, eff1) in _CARDS.items():
        for up, eff in ((0, eff0), (1, eff1)):
            r = cid * 2 + up
            cost[r], ctype[r], playable[r], exhaust[r] = c, t, p, e
            for k, row in enumerate(eff):
                effects[r, k] = row
    return cost, ctype, playable, exhaust, effects


CARD_COST, CARD_TYPE, CARD_PLAYABLE, CARD_EXHAUST, CARD_EFFECTS = _build()

# ---------------------------------------------------------------- The Champ (A0 numbers)
CHAMP_HP = 420
(CH_NONE, CH_HEAVY_SLASH, CH_DEFENSIVE_STANCE, CH_EXECUTE, CH_FACE_SLAP,
 CH_GLOAT, CH_TAUNT, CH_ANGER) = range(8)
