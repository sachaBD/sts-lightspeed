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


def retarget(s: State, move, t_card, t_up, t_cost, t_misc) -> State:
    """Move move[pile, d] copies of each slot d to the identity (t_card, t_up, t_cost, t_misc)[d], allocating
    slots as needed (never-used first, then slots with no copies left). Copies whose identity changes
    (an upgrade, a cost change) move this way. Sets ERR_SLOT_OVERFLOW if no slot is free."""
    D = s.slot_card.shape[0]
    t_card, t_up, t_cost, t_misc = (jnp.asarray(x, I32) for x in (t_card, t_up, t_cost, t_misc))
    moving = move.sum(0) > 0
    same = ((s.slot_card[None, :] == t_card[:, None]) & (s.slot_up[None, :] == t_up[:, None])
            & (s.slot_cost[None, :] == t_cost[:, None]) & (s.slot_misc[None, :] == t_misc[:, None])
            & (t_card[:, None] > 0))
    found = jnp.any(same, 1)
    target = jnp.argmax(same, 1)
    unused = jnp.all(s.piles == 0, axis=0)
    idx = jnp.arange(D)
    free_key = jnp.where(s.slot_card == 0, idx, jnp.where(unused & ~moving, D + idx, 3 * D))
    free_order = jnp.argsort(free_key)
    need = moving & ~found
    new_slot = free_order[jnp.clip(jnp.cumsum(need) - 1, 0, D - 1)]
    ok_new = free_key[new_slot] < 3 * D
    target = jnp.where(found, target, new_slot)
    write = jnp.where(need & ok_new, target, D)
    setv = lambda arr, v: arr.at[write].set(v.astype(I16), mode="drop")
    ok = found | ok_new
    m = jnp.where(ok[None, :], move, 0).astype(I32)
    p = s.piles.astype(I32) - m
    p = p.at[:, jnp.where(moving & ok, target, D)].add(m, mode="drop")
    return s._replace(slot_card=setv(s.slot_card, t_card), slot_up=setv(s.slot_up, t_up),
                      slot_cost=setv(s.slot_cost, t_cost), slot_misc=setv(s.slot_misc, t_misc),
                      piles=p.astype(I16),
                      error=s.error | jnp.where(jnp.any(moving & ~ok), U32(api.ERR_SLOT_OVERFLOW), U32(0)))


def _take_first(c, k):
    """Take up to k copies from counts c in slot order."""
    return jnp.clip(k - (jnp.cumsum(c) - c), 0, c)


def _sample(draw, disc, k, key, known_top=None):
    """k (<= MAX_DRAW) copies: the known top cards first, then uniformly without replacement from the draw pile,
    then (if it runs out) from the shuffled discard pile. One sample: known-top copies get keys -2, -1, other
    draw-pile copies keys in [0, 1), discard copies in [1, 2); the k smallest keys are drawn.
    Returns (drawn from draw [D], drawn from discard [D])."""
    D = draw.shape[0]
    counts = jnp.concatenate([draw, disc])
    cum = jnp.cumsum(counts)
    idx = jnp.arange(MAX_PILE)
    slot = jnp.searchsorted(cum, idx, side="right")         # which (pile, slot) copy i belongs to
    u = jax.random.uniform(key, (MAX_PILE,)) + (slot >= D)
    if known_top is not None:
        kt0, kt1 = known_top[0].astype(I32), known_top[1].astype(I32)
        rank = idx - (cum[jnp.minimum(slot, 2 * D - 1)] - counts[jnp.minimum(slot, 2 * D - 1)])  # copy number
        u = jnp.where((slot == kt0) & (rank == 0) & (kt0 >= 0), -2.0, u)
        u = jnp.where((slot == kt1) & (rank == jnp.where(kt1 == kt0, 1, 0)) & (kt1 >= 0), -1.0, u)
    u = jnp.where(idx < cum[-1], u, 3.0)
    _, chosen = jax.lax.top_k(-u, MAX_DRAW)                 # the MAX_DRAW smallest keys
    take = (jnp.arange(MAX_DRAW) < k).astype(I32)
    drawn = jnp.zeros(2 * D, I32).at[jnp.minimum(slot[chosen], 2 * D - 1)].add(take)
    return drawn[:D], drawn[D:]


def draw_cards(spec: api.EnvSpec, s: State, n, key, forced, status_first=True):
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
    kt = s.known_top.astype(I32)
    n_known = jnp.sum(kt >= 0)
    if forced is None:
        d1, d2 = _sample(draw, disc, k, key, kt)
    else:
        # the C++ outcome: known top cards first, then statuses (their triggers decide later draws) or, for the
        # earlier of a step's two draws, non-statuses first (the oracle can't tell which draw took a card)
        status = cards.rows(s)[:, cards.C("type")] == cards.tc.STATUS
        status = jnp.where(status_first, status, ~status)
        known = jnp.zeros(spec.D, I32).at[jnp.where(kt >= 0, kt, spec.D)].add(1, mode="drop")
        # forced counts follow cards whose identity changed during the step (cost changes): match by group =
        # same card, upgrade and misc; the group total is kept on its first slot
        group = ((s.slot_card[:, None] == s.slot_card[None, :]) & (s.slot_up[:, None] == s.slot_up[None, :])
                 & (s.slot_misc[:, None] == s.slot_misc[None, :]) & (s.slot_card[:, None] > 0))
        first = jnp.argmax(group, 1) == jnp.arange(spec.D)
        f_group = group.astype(I32) @ forced.draw.astype(I32)
        # known top cards are deterministic: drawn first whether or not the C++ outcome lists them
        t0 = _take_first(jnp.minimum(known, draw), k1)
        f = jnp.maximum(f_group - group.astype(I32) @ t0, 0)
        avail = jnp.minimum(f, draw - t0)
        t = _take_first(jnp.where(status, avail, 0), k1 - t0.sum())
        t = t0 + t + _take_first(jnp.where(status, 0, avail - t), k1 - t0.sum() - t.sum())
        d1 = jnp.where(k2 > 0, draw, t)
        rest = jnp.minimum(jnp.maximum(f - group.astype(I32) @ (d1 - t0), 0), disc)
        t = _take_first(jnp.where(status, rest, 0), k2)
        d2 = t + _take_first(jnp.where(status, 0, rest), k2 - t.sum())
        left = jnp.maximum(f - group.astype(I32) @ (d1 - t0 + d2), 0)
        forced = forced._replace(draw=jnp.where(first, left, 0).astype(forced.draw.dtype))
    shuffled = k2 > 0
    # Sundial: every 3rd shuffle gives 2 energy
    from jaxsim._core.tables import relics as tr
    sun = s.rel[tr.ID["SUNDIAL"]] & shuffled
    s = s._replace(r_sundial=jnp.where(sun, jnp.where(s.r_sundial == 2, 0, s.r_sundial + 1), s.r_sundial).astype(I16),
                   energy=(s.energy + 2 * (sun & (s.r_sundial == 2))).astype(I16))
    gone = jnp.minimum(k, n_known)                            # known top cards drawn: forget them
    kt_new = jnp.where(gone >= 2, jnp.full_like(kt, -1), jnp.where(gone == 1, jnp.stack([kt[1], -1]), kt))
    s = s._replace(known_top=kt_new.astype(I16))
    new_draw = jnp.where(shuffled, disc - d2, draw - d1)
    new_disc = jnp.where(shuffled, 0, disc)
    p = p.at[api.PILE_DRAW].set(new_draw).at[api.PILE_DISCARD].set(new_disc).at[api.PILE_HAND].add(d1 + d2)
    overflow = draw.sum() + disc.sum() > MAX_PILE
    s = s._replace(piles=p.astype(I16), error=s.error | jnp.where(overflow & go, U32(api.ERR_QUEUE_OVERFLOW), U32(0)))
    return s, forced, d1 + d2


def draw(spec: api.EnvSpec, s: State, n, key, forced, extend: bool, fire_now=True):
    """Draw n cards with on-draw triggers (CardManager::draw): each Status queues Evolve draws, each Status or
    Curse makes Fire Breathing damage the enemy. extend: Evolve draws follow in this draw (up to EVOLVE_ROUNDS);
    otherwise they are returned for a later draw. fire_now False: Fire Breathing damage is returned instead of
    dealt (end-of-turn Dark Embrace draws: it lands after the Champ's block reset).
    Returns (state, forced, Evolve draws not yet done, Fire Breathing damage deferred)."""
    from jaxsim._core import cards, hooks
    types = cards.rows(s)[:, cards.C("type")]
    keys = jax.random.split(key, EVOLVE_ROUNDS)
    pending = jnp.asarray(n, I32)
    later = jnp.zeros((), I32)
    deferred = jnp.zeros((), I32)
    for r in range(EVOLVE_ROUNDS if extend else 1):
        s, forced, drawn = draw_cards(spec, s, pending, keys[r], forced, status_first=extend | fire_now)
        statuses = jnp.sum(jnp.where(types == cards.tc.STATUS, drawn, 0))
        curses = jnp.sum(jnp.where(types == cards.tc.CURSE, drawn, 0))
        fb = (statuses + curses) * s.p_fire_breathing.astype(I32)
        s = hooks.hurt_enemy(s, fb, ~s.done & (fb > 0) & fire_now)
        deferred = deferred + jnp.where(fire_now, 0, fb)
        pending = statuses * s.p_evolve.astype(I32)
        if not extend:
            later = pending
    return s, forced, later, deferred
