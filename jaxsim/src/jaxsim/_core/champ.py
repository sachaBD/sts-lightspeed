"""The Champ at ascension 20: enemy turn, end of round and move AI as arithmetic.

Mirrors src/combat/MonsterSpecific.cpp (takeTurn THE_CHAMP_*, getMoveForRoll THE_CHAMP) and
BattleContext::afterMonsterTurns / applyEndOfRoundPowers.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, I32, State, tree_where

_M = {n: i for i, n in enumerate(api.MOVE_NAMES)}
HEAVY_SLASH = _M["THE_CHAMP_HEAVY_SLASH"]
DEFENSIVE_STANCE = _M["THE_CHAMP_DEFENSIVE_STANCE"]
EXECUTE = _M["THE_CHAMP_EXECUTE"]
FACE_SLAP = _M["THE_CHAMP_FACE_SLAP"]
GLOAT = _M["THE_CHAMP_GLOAT"]
TAUNT = _M["THE_CHAMP_TAUNT"]
ANGER = _M["THE_CHAMP_ANGER"]

HP = 440                 # A9+ fixed
HEAVY_SLASH_DMG = 18     # A4+
FACE_SLAP_DMG = 14       # A4+
EXECUTE_DMG, EXECUTE_HITS = 10, 2
GLOAT_STR = 5            # bossDiffIdx 2 (A19+)
ANGER_STR = 12
STANCE_BLOCK, STANCE_METALLICIZE = 20, 7   # A19+
STANCE_ROLL = 30         # A19+
JA_VULN, JA_WEAK, JA_FRAIL = 1, 2, 4
F32 = jnp.float32


def is_attack(move):
    """Monster::isAttacking for The Champ's moves (isMoveAttack)."""
    return (move == HEAVY_SLASH) | (move == FACE_SLAP) | (move == EXECUTE)


def roll(s: State, r) -> State:
    """getMoveForRoll: e_move is the move just taken (lastMove), e_last the one before (lastMoveBefore)."""
    last, before = s.e_move, s.e_last
    phase2 = (s.e_ai & 4) != 0
    below_half = s.e_hp < s.e_max_hp // 2
    execute = phase2 & (last != EXECUTE) & (before != EXECUTE)
    anger = ~phase2 & below_half
    taunt = ~phase2 & ~below_half & (((s.turn.astype(I32) + 2) % 4) == 0)
    stance = (r <= STANCE_ROLL) & (last != DEFENSIVE_STANCE) & ((s.e_ai & 3) < 2)
    gloat = (r <= 30) & (last != GLOAT) & (last != DEFENSIVE_STANCE)
    slap = (r <= 55) & (last != FACE_SLAP)
    heavy = last != HEAVY_SLASH
    move = jnp.select([execute, anger, taunt, stance, gloat, slap, heavy],
                      [EXECUTE, ANGER, TAUNT, DEFENSIVE_STANCE, GLOAT, FACE_SLAP, HEAVY_SLASH], FACE_SLAP)
    ai = s.e_ai | jnp.where(anger, 4, 0)
    ai = ai + jnp.where(move == DEFENSIVE_STANCE, 1, 0)
    return s._replace(e_last=last, e_move=move.astype(I16), e_ai=ai.astype(I16))


def monster_damage(s: State, base):
    """Monster::calculateDamageToPlayer."""
    d = (base + s.e_str.astype(I32)).astype(F32)
    d = jnp.where(s.e_weak > 0, d * F32(0.75), d)
    d = jnp.where(s.p_vuln > 0, d * F32(1.5), d)
    return jnp.maximum(d.astype(I32), 0)


def _debuff_player(s: State, field: str, bit: int, amount) -> State:
    """Player::debuff from a monster: just-applied only when the debuff is new (C++ `!hasStatus`)."""
    cur = getattr(s, field)
    new = (amount > 0) & (cur == 0)
    return s._replace(**{field: (cur + amount).astype(I16)}, p_ja=(s.p_ja | jnp.where(new, bit, 0)).astype(I16))


def take_turn(s: State, r, jugg_late=0, fb_late=0) -> State:
    """The Champ's move, hit by hit: block absorbs, hp loss (Blood for Blood), Flame Barrier hits back after
    each hit. Nothing more happens once the fight is over."""
    from jaxsim._core import hooks
    i16 = hooks.i16
    m = s.e_move
    s = s._replace(e_block=jnp.where(s.done, s.e_block, I16(0)))
    # queued behind the monster turn start, so after the block reset: Fire Breathing from end-of-turn Dark
    # Embrace draws, then Juggernaut from the end-of-turn Feel No Pain block
    s = hooks.hurt_enemy(s, jnp.asarray(fb_late, I32), ~s.done & (jnp.asarray(fb_late) > 0))
    s = hooks.juggernaut(s, jugg_late)
    base = jnp.select([m == HEAVY_SLASH, m == FACE_SLAP, m == EXECUTE], [HEAVY_SLASH_DMG, FACE_SLAP_DMG, EXECUTE_DMG], 0)
    hits = jnp.select([m == EXECUTE, base > 0], [EXECUTE_HITS, 1], 0)
    per = monster_damage(s, base)
    for h in range(EXECUTE_HITS):
        hit = ~s.done & (h < hits)
        blocked = jnp.minimum(s.block.astype(I32), jnp.where(hit, per, 0))
        s = s._replace(block=i16(s.block - blocked))
        s = hooks.lose_hp(s, jnp.where(hit, per - blocked, 0), False, hit)
        s = hooks.hurt_enemy(s, s.p_flame_barrier.astype(I32), hit & ~s.done & (s.p_flame_barrier > 0))
    go = ~s.done
    slap, taunt = (m == FACE_SLAP) & go, (m == TAUNT) & go
    s = _debuff_player(s, "p_frail", JA_FRAIL, jnp.where(slap, 2, 0))
    s = _debuff_player(s, "p_vuln", JA_VULN, jnp.where(slap | taunt, 2, 0))
    s = _debuff_player(s, "p_weak", JA_WEAK, jnp.where(taunt, 2, 0))
    anger = (m == ANGER) & go
    e_str = jnp.where(anger, jnp.maximum(s.e_str, 0), s.e_str)
    e_str = e_str + jnp.select([anger, (m == GLOAT) & go], [ANGER_STR, GLOAT_STR], 0)
    stance = (m == DEFENSIVE_STANCE) & go
    s = s._replace(e_str=i16(e_str), e_vuln=i16(jnp.where(anger, 0, s.e_vuln)), e_weak=i16(jnp.where(anger, 0, s.e_weak)),
                   e_block=i16(s.e_block + jnp.where(stance, STANCE_BLOCK, 0)),
                   e_metallicize=i16(s.e_metallicize + jnp.where(stance, STANCE_METALLICIZE, 0)))
    rolled = roll(s, r)
    return s._replace(e_move=jnp.where(go, rolled.e_move, s.e_move), e_last=jnp.where(go, rolled.e_last, s.e_last),
                      e_ai=jnp.where(go, rolled.e_ai, s.e_ai))


def end_of_round(s: State) -> State:
    """applyEndOfRoundPowers: monster end-of-turn triggers, player end of round, monster end of round."""
    i16 = lambda x: jnp.asarray(x).astype(I16)
    dec = lambda v, bit: jnp.where((s.p_ja & bit) != 0, v, jnp.maximum(v - 1, 0))
    return s._replace(
        e_block=i16(s.e_block + s.e_metallicize),
        p_vuln=i16(dec(s.p_vuln, JA_VULN)), p_weak=i16(dec(s.p_weak, JA_WEAK)), p_frail=i16(dec(s.p_frail, JA_FRAIL)),
        p_ja=I16(0),
        e_vuln=i16(jnp.maximum(s.e_vuln - 1, 0)), e_weak=i16(jnp.maximum(s.e_weak - 1, 0)))
