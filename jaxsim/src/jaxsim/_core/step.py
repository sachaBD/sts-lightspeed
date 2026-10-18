"""step and legal_mask.

One step = one decision. Both decision kinds are computed and selected (branch-free under vmap), around two
shared draws that follow the C++ action-queue order:

    play card:  play_pre  -> draw A (the card's own draws) -> play_post (card leaves, exhaust triggers,
                Juggernaut) -> draw B (Dark Embrace / Evolve draws)
    END_TURN:   end_pre (Metallicize, Burn, Combust, end-of-turn powers, ethereal exhaust, discard)
                -> draw A (Dark Embrace draws from ethereal exhaust) -> end_mid (Champ turn, end of round,
                next turn start, Brutality) -> draw B (5 + Brutality + Evolve) -> Demon Form, energy

Ethereal-exhaust Dark Embrace draws trigger Evolve after the Champ's turn here (the C++ queues those Evolve
draws behind the monster turn start); only Fire Breathing timing on such a draw could differ.
"""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import cards, champ, hooks, piles
from jaxsim._core.state import I16, I32, U32, State, tree_where
from jaxsim._core.tables import cards as tc

HAND_DRAW = 5
MAX_DRAW = 10   # per draw point: the hand holds at most 10 cards
i16 = hooks.i16


def legal_mask(spec: api.EnvSpec, s: State):
    mask = jnp.zeros((spec.A,), bool)
    mask = mask.at[: spec.D].set(cards.playable(s) & ~s.done)
    return mask.at[spec.END_TURN].set(True)


def end_pre(spec: api.EnvSpec, s: State):
    """Up to the discard. Returns (state, Dark Embrace draws)."""
    r = cards.rows(s)
    # Metallicize block (Juggernaut)
    s, jugg = hooks.gain_block(s, s.p_metallicize.astype(I32), ~s.done)
    # end-of-turn cards in hand (Burn): blockable self damage, one hit per Burn
    hand = s.piles[api.PILE_HAND].astype(I32)
    burns = jnp.where(r[:, tc.col("eot_damage")] > 0, hand, 0)
    per_burn = jnp.max(jnp.where(burns > 0, r[:, tc.col("eot_damage")], 0))   # Burn+ only from other enemies
    s = hooks.damage_player_hits(s, per_burn, burns.sum(), True, ~s.done)
    # Combust: lose 1 hp per Combust played (self damage), then damage the enemy
    s = hooks.lose_hp(s, s.p_combust_hp.astype(I32), True, ~s.done & (s.p_combust > 0))
    s = hooks.hurt_enemy(s, s.p_combust.astype(I32), ~s.done & (s.p_combust > 0))
    # end-of-turn powers: Flex's strength loss, No Draw and Rage expire
    s = s._replace(p_str=i16(s.p_str - s.p_lose_str), p_lose_str=I16(0), p_no_draw=I16(0), p_rage=I16(0))
    # ethereal cards exhaust (with triggers), the rest of the hand is discarded
    s, _, j2, de = cards.exhaust_hand(s, r[:, tc.col("ethereal")] == 1, ~s.done)
    p = s.piles.astype(I32)
    p = p.at[api.PILE_DISCARD].add(p[api.PILE_HAND]).at[api.PILE_HAND].set(0)
    s = s._replace(piles=p.astype(I16))
    s = hooks.juggernaut(s, jugg + j2)
    return s, de


def end_mid(spec: api.EnvSpec, s: State, roll):
    """Champ turn, end of round, next turn start. Returns (state, cards to draw)."""
    s = champ.take_turn(s, roll)
    go = ~s.done
    nxt = champ.end_of_round(s)
    turn = nxt.turn + 1
    cap = turn >= spec.max_turns
    block = jnp.where(nxt.p_barricade > 0, nxt.block, I16(0))
    nxt = nxt._replace(turn=i16(turn), block=block, p_flame_barrier=I16(0), done=nxt.done | cap,
                       truncated=nxt.truncated | cap)
    s = tree_where(go, nxt, s)
    # Brutality: lose hp (not self damage) and draw, after the hand draw
    s = hooks.lose_hp(s, s.p_brutality.astype(I32), False, ~s.done & (s.p_brutality > 0))
    return s, jnp.where(s.done, 0, HAND_DRAW + s.p_brutality.astype(I32))


def end_post(s: State) -> State:
    go = ~s.done
    return s._replace(p_str=i16(s.p_str + jnp.where(go, s.p_demon_form, 0)),
                      energy=jnp.where(go, s.p_energy_per_turn, s.energy))


def step(spec: api.EnvSpec, s: State, action, key, forced=None) -> State:
    action = jnp.asarray(action, I32)
    a = jnp.clip(action, 0, spec.A - 1)
    ok = (action >= 0) & (action < spec.A) & legal_mask(spec, s)[a]
    is_end = a == spec.END_TURN
    d = jnp.clip(a, 0, spec.D - 1)
    k_a, k_b, k_roll = jax.random.split(key, 3)
    roll = jax.random.randint(k_roll, (), 0, 100) if forced is None else forced.roll

    played, n_own, trig, jugg = cards.play_pre(spec, s, d)
    ended, n_de = end_pre(spec, s)
    mid = tree_where(is_end, ended, played)
    mid, forced, evolve_later = piles.draw(spec, mid, jnp.where(is_end, n_de, n_own), k_a, forced, MAX_DRAW, extend=False)

    left, trig2 = cards.play_post(spec, mid, d, jugg)
    turned, n_hand = end_mid(spec, mid, roll)
    mid = tree_where(is_end, turned, left)
    n_b = jnp.where(is_end, n_hand, trig + trig2) + evolve_later
    mid, _, _ = piles.draw(spec, mid, jnp.where(mid.done, 0, n_b), k_b, forced, MAX_DRAW, extend=True)
    nxt = tree_where(is_end, end_post(mid), mid)

    # one full select: illegal actions and terminal states keep s; then patch flags
    take = ok & ~s.done
    out = tree_where(take, nxt, s)
    bad = ~ok & ~s.done
    return out._replace(done=out.done | bad, truncated=out.truncated | bad,
                        error=out.error | jnp.where(bad, U32(api.ERR_ILLEGAL_ACTION), U32(0)))
