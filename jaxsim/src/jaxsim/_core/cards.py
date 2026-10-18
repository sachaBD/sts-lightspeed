"""Card play: legality and the fused resolve over the card table's fixed effect columns.

A play is split around the shared draw in step.py: play_pre (hp loss, block, damage, debuffs, buffs, energy)
returns the number of cards to draw; play_post (created cards, Battle Trance, the played card leaving) runs
after it. See tables/cards.py for the effect order.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import piles
from jaxsim._core.state import I16, I32, State, tree_where
from jaxsim._core.tables import cards as tc
from jaxsim._core import champ

TABLE = jnp.asarray(tc.TABLE)
F32 = jnp.float32
C = tc.col


def rows(s: State):
    """[D, n_columns] table rows of every slot."""
    return TABLE[s.slot_card.astype(I32), jnp.minimum(s.slot_up, 1).astype(I32)]


def row(s: State, d):
    return TABLE[s.slot_card[d].astype(I32), jnp.minimum(s.slot_up[d], 1).astype(I32)]


def playable(s: State):
    r = rows(s)
    cost = s.slot_cost.astype(I32)
    hand = s.piles[api.PILE_HAND] > 0
    only_attacks = jnp.sum(jnp.where(hand & (r[:, C("type")] != tc.ATTACK), 1, 0)) == 0
    clash_ok = (r[:, C("special")] != tc.SP_CLASH) | only_attacks
    return hand & (r[:, C("playable")] == 1) & (cost >= 0) & (cost <= s.energy) & clash_ok


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
    r = row(s, d)
    sp = r[C("special")]
    i16 = lambda x: jnp.asarray(x).astype(I16)
    hand = s.piles[api.PILE_HAND]
    s = s._replace(energy=i16(s.energy - s.slot_cost[d]), piles=s.piles.at[api.PILE_HAND, d].set(hand[d] - 1))

    # self hp loss (no block); dying here ends the fight
    hp = s.hp - r[C("hp_loss")]
    died = hp <= 0
    s = s._replace(hp=i16(jnp.maximum(hp, 0)), done=s.done | died)
    alive = ~s.done

    # block (Entrench doubles current block, without dex/frail)
    gain = jnp.where(r[C("block")] > 0, card_block(s, r[C("block")]), 0)
    gain = gain + jnp.where(sp == tc.SP_ENTRENCH, s.block, 0)
    s = s._replace(block=i16(s.block + gain * alive))

    # damage x hits (same per-hit damage; block absorbs hit by hit, which sums to this)
    dropkick_bonus = (sp == tc.SP_DROPKICK) & (s.e_vuln > 0)          # checked before the hit
    total = player_attack_damage(s, r[C("damage")]) * r[C("hits")] * alive
    blocked = jnp.minimum(s.e_block.astype(I32), total)
    unblocked = total - blocked
    e_hp = s.e_hp - unblocked
    won = e_hp <= 0
    hp_lost = jnp.minimum(unblocked, s.e_hp)
    s = s._replace(e_block=i16(s.e_block - blocked), e_hp=i16(jnp.maximum(e_hp, 0)), won=s.won | won,
                   done=s.done | won)
    # Reaper heals the hp it removed; Feed raises max hp on the kill (both survive the win)
    heal = jnp.where(sp == tc.SP_REAPER, hp_lost, 0)
    feed = jnp.where((sp == tc.SP_FEED) & won, r[C("special_amt")], 0)
    max_hp = s.max_hp + feed
    s = s._replace(max_hp=i16(max_hp), hp=i16(jnp.minimum(s.hp + heal + feed, max_hp)))

    # everything below is cleared on victory
    live = ~s.done
    p_str = s.p_str + r[C("p_str")] * live
    p_str = p_str + jnp.where((sp == tc.SP_SPOT_WEAKNESS) & champ.is_attack(s.e_move) & live, r[C("special_amt")], 0)
    p_str = jnp.where((sp == tc.SP_LIMIT_BREAK) & live, p_str * 2, p_str)
    s = s._replace(e_vuln=i16(s.e_vuln + r[C("e_vuln")] * live), e_weak=i16(s.e_weak + r[C("e_weak")] * live),
                   e_str=i16(s.e_str + r[C("e_str")] * live), p_str=i16(p_str),
                   p_lose_str=i16(s.p_lose_str + r[C("lose_str")] * live),
                   energy=i16(s.energy + (r[C("energy")] + dropkick_bonus) * live))
    n_draw = (r[C("draw")] + dropkick_bonus) * live * (s.p_no_draw == 0)
    return s, n_draw


def play_post(spec: api.EnvSpec, s: State, d):
    """Created cards and Battle Trance (cleared on victory), then the played card leaves (unless the player died)."""
    r = row(s, d)
    live = ~s.done
    p = s.piles.astype(I32)
    p = p.at[api.PILE_DISCARD, d].add(r[C("copy_to_discard")] * live)
    s = s._replace(piles=p.astype(I16), p_no_draw=(s.p_no_draw + ((r[C("special")] == tc.SP_BATTLE_TRANCE) & live))
                                                             .astype(I16))
    # created cards: draw pile (shuffled in), discard pile, or hand (overflowing to discard at 10)
    card, pile, n = r[C("create_card")], r[C("create_pile")], r[C("create_n")]
    s2, slot, ok = piles.find_or_alloc(s, card, 0, TABLE[card, 0, C("cost")], 0)
    p = s2.piles.astype(I32)
    room = jnp.maximum(spec.H - p[api.PILE_HAND].sum(), 0)
    to_hand = jnp.where(pile == tc.TO_HAND, jnp.minimum(n, room), 0)
    to_discard = jnp.where(pile == tc.TO_DISCARD, n, 0) + jnp.where(pile == tc.TO_HAND, n - to_hand, 0)
    to_draw = jnp.where(pile == tc.TO_DRAW, n, 0)
    p = (p.at[api.PILE_HAND, slot].add(to_hand).at[api.PILE_DISCARD, slot].add(to_discard)
          .at[api.PILE_DRAW, slot].add(to_draw))
    s = tree_where((card > 0) & live & ok, s2._replace(piles=p.astype(I16)), s)
    # the played card leaves: powers vanish, exhausting cards go to exhaust, the rest to discard.
    # After a win it still leaves (OnAfterCardUsed survives victory); after the player dies it stays in limbo.
    dest = jnp.where(r[C("exhausts")] == 1, api.PILE_EXHAUST, api.PILE_DISCARD)
    p = s.piles.astype(I32)
    p = p.at[dest, d].add(jnp.where((r[C("type")] == tc.POWER) | (s.hp <= 0), 0, 1))
    return s._replace(piles=p.astype(I16))
