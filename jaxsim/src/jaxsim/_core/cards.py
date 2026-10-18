"""Card play: legality and the fused resolve over the card table's fixed effect columns.

A play is split around the shared draw in step.py: play_pre (energy, block, damage, debuffs, buffs) returns the
number of cards to draw, play_post (created cards, the played card leaving the hand area) runs after it.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import piles
from jaxsim._core.state import I16, I32, State, tree_where
from jaxsim._core.tables import cards as tc

TABLE = jnp.asarray(tc.TABLE)
F32 = jnp.float32
C = tc.col


def rows(s: State):
    """[D, n_columns] table rows of every slot."""
    return TABLE[s.slot_card.astype(I32), jnp.minimum(s.slot_up, 1).astype(I32)]


def playable(s: State):
    r = rows(s)
    cost = s.slot_cost.astype(I32)
    return (s.piles[api.PILE_HAND] > 0) & (r[:, C("playable")] == 1) & (cost >= 0) & (cost <= s.energy)


def player_attack_damage(s: State, base):
    """BattleContext::calculateCardDamage: float32 chain, truncated once."""
    d = base.astype(F32) + s.p_str.astype(F32)
    d = jnp.where(s.p_weak > 0, d * F32(0.75), d)
    d = jnp.where(s.e_vuln > 0, d * F32(1.5), d)
    return jnp.maximum(d.astype(I32), 0)


def card_block(s: State, base):
    """BattleContext::calculateCardBlock: integer maths."""
    b = jnp.where(s.p_dex != 0, jnp.maximum(base + s.p_dex.astype(I32), 0), base)
    return jnp.where(s.p_frail > 0, b * 3 // 4, b)


def play_pre(spec: api.EnvSpec, s: State, d):
    r = TABLE[s.slot_card[d].astype(I32), jnp.minimum(s.slot_up[d], 1).astype(I32)]
    i16 = lambda x: x.astype(I16)
    hand = s.piles[api.PILE_HAND]
    s = s._replace(energy=i16(s.energy - s.slot_cost[d]), piles=s.piles.at[api.PILE_HAND, d].set(hand[d] - 1))
    # block
    once = card_block(s, r[C("block")])
    gain = jnp.where(r[C("block_twice")] == 1, card_block(s, once), once)
    block = s.block + jnp.where(r[C("block")] > 0, gain, 0)
    s = s._replace(block=i16(block))
    # damage x hits (same per-hit damage; block absorbs hit by hit, which sums to this)
    total = player_attack_damage(s, r[C("damage")]) * r[C("hits")]
    blocked = jnp.minimum(s.e_block.astype(I32), total)
    e_hp = s.e_hp - (total - blocked)
    won = e_hp <= 0
    s = s._replace(e_block=i16(s.e_block - blocked), e_hp=i16(jnp.maximum(e_hp, 0)), won=s.won | won, done=s.done | won)
    # debuffs, buffs: queued after the damage, so skipped if it won the fight
    live = ~s.done
    s = s._replace(e_vuln=i16(s.e_vuln + r[C("e_vuln")] * live), e_weak=i16(s.e_weak + r[C("e_weak")] * live),
                   p_str=i16(s.p_str + r[C("p_str")] * live))
    return s, r[C("draw")]


def play_post(spec: api.EnvSpec, s: State, d):
    """Queued effects after the draw are skipped once the fight is over; the played card still leaves
    (C++ OnAfterCardUsed survives combat end)."""
    r = TABLE[s.slot_card[d].astype(I32), jnp.minimum(s.slot_up[d], 1).astype(I32)]
    live = ~s.done
    p = s.piles.astype(I32)
    p = p.at[api.PILE_DISCARD, d].add(r[C("copy_to_discard")] * live)
    s = s._replace(piles=p.astype(I16))
    # a created card shuffled into the draw pile (Wild Strike's Wound)
    add = r[C("add_draw_card")]
    new_cost = TABLE[add, 0, C("cost")]
    s2, slot, ok = piles.find_or_alloc(s, add, 0, new_cost, 0)
    s2 = s2._replace(piles=s2.piles.at[api.PILE_DRAW, slot].add(ok.astype(I16)))
    s = tree_where((add > 0) & live, s2, s)
    # the played card leaves: powers vanish, exhausting cards go to exhaust, the rest to discard
    is_power = r[C("type")] == tc.POWER
    dest = jnp.where(r[C("exhausts")] == 1, api.PILE_EXHAUST, api.PILE_DISCARD)
    p = s.piles.astype(I32)
    p = p.at[dest, d].add(jnp.where(is_power, 0, 1))
    return s._replace(piles=p.astype(I16))

