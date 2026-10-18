"""Pile counts: draw (sample from counts, reshuffling the discard pile when empty) and slot allocation."""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, I32, U32, State


def find_or_alloc(s: State, card, up, cost, misc):
    """Slot holding this identity, else the first empty slot. Sets ERR_SLOT_OVERFLOW if neither exists."""
    match = (s.slot_card == card) & (s.slot_up == up) & (s.slot_cost == cost) & (s.slot_misc == misc)
    unused = jnp.all(s.piles == 0, axis=0) & ~jnp.any(s.known_top[:, None] == jnp.arange(s.slot_card.shape[0]), 0)
    # never-used slots first (deterministic for the parity harness), then slots with no copies left
    empty = jnp.where(jnp.any(s.slot_card == 0), s.slot_card == 0, unused)
    have = jnp.any(match)
    ok = have | jnp.any(empty)
    slot = jnp.where(have, jnp.argmax(match), jnp.argmax(empty))
    setv = lambda arr, v: arr.at[slot].set(jnp.where(ok, jnp.asarray(v, I16), arr[slot]))
    s = s._replace(slot_card=setv(s.slot_card, card), slot_up=setv(s.slot_up, up),
                   slot_cost=setv(s.slot_cost, cost), slot_misc=setv(s.slot_misc, misc),
                   error=s.error | jnp.where(ok, U32(0), U32(api.ERR_SLOT_OVERFLOW)))
    return s, slot, ok


def _draw_one(spec: api.EnvSpec, s: State, active, key, forced):
    piles = s.piles.astype(I32)
    draw, disc, hand = piles[api.PILE_DRAW], piles[api.PILE_DISCARD], piles[api.PILE_HAND]
    shuffle = active & (draw.sum() == 0) & (disc.sum() > 0)
    draw = jnp.where(shuffle, draw + disc, draw)
    disc = jnp.where(shuffle, 0, disc)
    ok = active & (draw.sum() > 0) & (hand.sum() < spec.H)
    if forced is None:
        r = jax.random.randint(key, (), 0, jnp.maximum(draw.sum(), 1))
        slot = jnp.sum(jnp.cumsum(draw) <= r)
    else:
        avail = (forced.draw > 0) & (draw > 0)
        ok = ok & jnp.any(avail)
        slot = jnp.argmax(avail)
        forced = forced._replace(draw=forced.draw - ((jnp.arange(spec.D) == slot) & ok))
    take = ((jnp.arange(spec.D) == slot) & ok).astype(I32)
    piles = piles.at[api.PILE_DRAW].set(draw - take).at[api.PILE_DISCARD].set(disc).at[api.PILE_HAND].set(hand + take)
    return s._replace(piles=piles.astype(I16)), forced


def draw(spec: api.EnvSpec, s: State, n, key, forced, max_n: int):
    """Draw n (traced, <= max_n) cards one at a time; stops at an empty draw+discard or a full hand."""
    keys = jax.random.split(key, max_n)
    for i in range(max_n):
        s, forced = _draw_one(spec, s, i < n, keys[i], forced)
    return s, forced
