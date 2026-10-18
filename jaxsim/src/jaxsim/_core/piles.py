"""Pile counts: drawing (sampled from counts, reshuffling the discard pile when empty) and slot allocation.

Drawing k cards from a uniformly shuffled pile is the same as choosing k of its cards uniformly without
replacement, so a draw is one vectorised sample (each card copy gets a random key; the k smallest are drawn)
instead of a loop of single draws. On-draw triggers (Evolve, Fire Breathing) are computed from what was drawn.
"""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, I32, U32, State

MAX_PILE = 128      # card copies in draw + discard that a draw can reach (more sets ERR_QUEUE_OVERFLOW)
MAX_DRAW = 10       # cards per draw: the hand holds at most 10
EVOLVE_ROUNDS = 10  # a draw plus up to 9 Evolve rounds: each round adds >= 1 card and the hand holds 10


def find_or_alloc(s: State, card, up, cost, misc, want=True):
    """Slot holding this identity, else a free slot; allocated only where `want`. Sets ERR_SLOT_OVERFLOW if
    a wanted identity has no slot. Returns (state, slot, ok)."""
    match = (s.slot_card == card) & (s.slot_up == up) & (s.slot_cost == cost) & (s.slot_misc == misc)
    unused = jnp.all(s.piles == 0, axis=0) & ~jnp.any(s.known_top[:, None] == jnp.arange(s.slot_card.shape[0]), 0)
    # never-used slots first (deterministic for the parity harness), then slots with no copies left
    empty = jnp.where(jnp.any(s.slot_card == 0), s.slot_card == 0, unused)
    have = jnp.any(match)
    ok = have | jnp.any(empty)
    slot = jnp.where(have, jnp.argmax(match), jnp.argmax(empty))
    write = want & ok
    setv = lambda arr, v: arr.at[slot].set(jnp.where(write, jnp.asarray(v, I16), arr[slot]))
    s = s._replace(slot_card=setv(s.slot_card, card), slot_up=setv(s.slot_up, up),
                   slot_cost=setv(s.slot_cost, cost), slot_misc=setv(s.slot_misc, misc),
                   error=s.error | jnp.where(want & ~ok, U32(api.ERR_SLOT_OVERFLOW), U32(0)))
    return s, slot, ok


def _take_first(c, k):
    """Take up to k copies from counts c in slot order."""
    return jnp.clip(k - (jnp.cumsum(c) - c), 0, c)


def _sample(draw, disc, k, key):
    """k (<= MAX_DRAW) copies: uniformly without replacement from the draw pile, then (if it runs out) from
    the shuffled discard pile. One sample: draw-pile copies get keys in [0, 1), discard copies in [1, 2), and
    the k smallest keys are drawn. Returns (drawn from draw [D], drawn from discard [D])."""
    D = draw.shape[0]
    counts = jnp.concatenate([draw, disc])
    cum = jnp.cumsum(counts)
    idx = jnp.arange(MAX_PILE)
    slot = jnp.searchsorted(cum, idx, side="right")         # which (pile, slot) copy i belongs to
    u = jax.random.uniform(key, (MAX_PILE,)) + (slot >= D)
    u = jnp.where(idx < cum[-1], u, 3.0)
    _, chosen = jax.lax.top_k(-u, MAX_DRAW)                 # the MAX_DRAW smallest keys
    take = (jnp.arange(MAX_DRAW) < k).astype(I32)
    drawn = jnp.zeros(2 * D, I32).at[jnp.minimum(slot[chosen], 2 * D - 1)].add(take)
    return drawn[:D], drawn[D:]


def draw_cards(spec: api.EnvSpec, s: State, n, key, forced):
    """BattleContext::drawCards for n cards: stops at a full hand, No Draw or the end of the fight; when the
    draw pile runs out the discard pile is shuffled in and the rest is drawn from it.
    Returns (state, forced, drawn counts [D])."""
    from jaxsim._core import cards
    p = s.piles.astype(I32)
    draw, disc, hand = p[api.PILE_DRAW], p[api.PILE_DISCARD], p[api.PILE_HAND]
    go = ~s.done & (s.p_no_draw == 0)
    k = jnp.where(go, jnp.minimum(jnp.minimum(n, spec.H - hand.sum()), draw.sum() + disc.sum()), 0)
    k = jnp.maximum(k, 0)
    k1 = jnp.minimum(k, draw.sum())
    k2 = k - k1                                              # > 0: the draw pile empties, then reshuffle
    if forced is None:
        d1, d2 = _sample(draw, disc, k, key)
    else:
        # the C++ outcome: statuses first (their triggers decide later draws), then slot order
        status = cards.rows(s)[:, cards.C("type")] == cards.tc.STATUS
        f = forced.draw.astype(I32)
        avail = jnp.minimum(f, draw)
        t = _take_first(jnp.where(status, avail, 0), k1)
        d1 = jnp.where(k2 > 0, draw, t + _take_first(jnp.where(status, 0, avail), k1 - t.sum()))
        rest = jnp.minimum(f - d1, disc)
        t = _take_first(jnp.where(status, rest, 0), k2)
        d2 = t + _take_first(jnp.where(status, 0, rest), k2 - t.sum())
        forced = forced._replace(draw=(f - d1 - d2).astype(forced.draw.dtype))
    shuffled = k2 > 0
    new_draw = jnp.where(shuffled, disc - d2, draw - d1)
    new_disc = jnp.where(shuffled, 0, disc)
    p = p.at[api.PILE_DRAW].set(new_draw).at[api.PILE_DISCARD].set(new_disc).at[api.PILE_HAND].add(d1 + d2)
    overflow = draw.sum() + disc.sum() > MAX_PILE
    s = s._replace(piles=p.astype(I16), error=s.error | jnp.where(overflow & go, U32(api.ERR_QUEUE_OVERFLOW), U32(0)))
    return s, forced, d1 + d2


def draw(spec: api.EnvSpec, s: State, n, key, forced, extend: bool):
    """Draw n cards with on-draw triggers (CardManager::draw): each Status queues Evolve draws, each Status or
    Curse makes Fire Breathing damage the enemy. extend: Evolve draws follow in this draw (up to EVOLVE_ROUNDS);
    otherwise they are returned for a later draw. Returns (state, forced, Evolve draws not yet done)."""
    from jaxsim._core import cards, hooks
    types = cards.rows(s)[:, cards.C("type")]
    keys = jax.random.split(key, EVOLVE_ROUNDS)
    pending = jnp.asarray(n, I32)
    later = jnp.zeros((), I32)
    for r in range(EVOLVE_ROUNDS if extend else 1):
        s, forced, drawn = draw_cards(spec, s, pending, keys[r], forced)
        statuses = jnp.sum(jnp.where(types == cards.tc.STATUS, drawn, 0))
        curses = jnp.sum(jnp.where(types == cards.tc.CURSE, drawn, 0))
        s = hooks.hurt_enemy(s, (statuses + curses) * s.p_fire_breathing.astype(I32),
                             ~s.done & (s.p_fire_breathing > 0) & (statuses + curses > 0))
        pending = statuses * s.p_evolve.astype(I32)
        if not extend:
            later = pending
    return s, forced, later
