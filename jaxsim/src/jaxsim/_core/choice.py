"""Pending-choice mode: single-card picks from one pile (Armaments, Headbutt, True Grit+, Burning Pact, Warcry,
Dual Wield, Exhume).

State.choice = [kind, limbo slot (the card being played), Juggernaut events pending, triggered draws pending].
kind 0 = no choice. Like the C++, a choice with 0 valid copies does nothing and one with exactly 1 valid copy
resolves itself; only with 2+ valid copies does the step end with the choice pending. The rest of the card
(its later draws, the card leaving play, queued triggers) then runs in the pick step.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import hooks
from jaxsim._core.state import I16, I32, State
from jaxsim._core.tables import cards as tc

i16 = hooks.i16
KIND, LIMBO, JUGG, TRIG = range(4)


def _rows(s: State):
    from jaxsim._core import cards
    return cards.rows(s)


def can_upgrade(s: State):
    """CardInstance::canUpgrade per slot: not yet upgraded (Searing Blow always), not a curse or status."""
    r = _rows(s)
    searing = s.slot_card == tc._ID["SEARING_BLOW"]
    t = r[:, tc.col("type")]
    return ((s.slot_up == 0) | searing) & (t != tc.CURSE) & (t != tc.STATUS) & (s.slot_card != 0)


def valid(s: State, kind):
    """[D] copies that may be picked for this kind."""
    p = s.piles.astype(I32)
    hand, disc, exh = p[api.PILE_HAND], p[api.PILE_DISCARD], p[api.PILE_EXHAUST]
    t = _rows(s)[:, tc.col("type")]
    exhume_ok = hand.sum() < spec_hand(s)
    return jnp.select(
        [kind == tc.CH_ARMAMENTS, kind == tc.CH_HEADBUTT, (kind == tc.CH_EXHAUST_ONE) | (kind == tc.CH_WARCRY),
         kind == tc.CH_DUAL_WIELD, kind == tc.CH_EXHUME],
        [jnp.where(can_upgrade(s), hand, 0), disc, hand,
         jnp.where((t == tc.ATTACK) | (t == tc.POWER), hand, 0),
         jnp.where((s.slot_card != tc._ID["EXHUME"]) & exhume_ok, exh, 0)],
        jnp.zeros_like(hand))


def spec_hand(s: State):
    return 10


def upgrade_copies(s: State, n) -> State:
    """Move n[d] hand copies of each slot d to its upgraded identity (CardInstance::upgrade). Cost: the upgraded
    table cost if upgrading changes the base cost, else the current cost."""
    from jaxsim._core import cards, piles
    card = s.slot_card.astype(I32)
    up0 = jnp.minimum(s.slot_up, 1).astype(I32)
    searing = card == tc._ID["SEARING_BLOW"]
    t_up = jnp.where(searing, s.slot_up + 1, 1)
    c0, c1 = cards.TABLE[card, 0, tc.col("cost")], cards.TABLE[card, 1, tc.col("cost")]
    t_cost = jnp.where((c0 != c1) & (up0 == 0), c1, s.slot_cost)
    move = jnp.zeros_like(s.piles, I32).at[api.PILE_HAND].set(jnp.asarray(n, I32))
    return piles.retarget(s, move, card, t_up, t_cost, s.slot_misc)


def push_top(s: State, slot, go) -> State:
    """A card put on top of the draw pile becomes known."""
    kt = jnp.where(go, jnp.concatenate([jnp.asarray(slot, I16)[None], s.known_top[:-1]]), s.known_top)
    return s._replace(known_top=kt)


def apply_pick(spec: api.EnvSpec, s: State, kind, d, limbo, go):
    """The effect of picking slot d. Returns (state, Juggernaut events, triggered draws)."""
    from jaxsim._core import cards
    one = (jnp.arange(spec.D) == d) & go
    p = s.piles.astype(I32)
    hand_kinds = (kind == tc.CH_EXHAUST_ONE) | (kind == tc.CH_WARCRY)
    # Armaments: upgrade the copy
    s = upgrade_copies(s, jnp.where((kind == tc.CH_ARMAMENTS) & one, 1, 0))
    p = s.piles.astype(I32)
    # moves between piles
    from_hand = jnp.where(hand_kinds & one, 1, 0)
    from_disc = jnp.where((kind == tc.CH_HEADBUTT) & one, 1, 0)
    from_exh = jnp.where((kind == tc.CH_EXHUME) & one, 1, 0)
    to_top = from_disc + jnp.where((kind == tc.CH_WARCRY) & one, 1, 0)
    to_exh = jnp.where((kind == tc.CH_EXHAUST_ONE) & one, 1, 0)
    # Dual Wield: copies of the picked card into the hand (overflow to discard)
    copies = 1 + jnp.minimum(s.slot_up[limbo], 1).astype(I32)
    room = jnp.maximum(spec.H - p[api.PILE_HAND].sum(), 0)
    dw = (kind == tc.CH_DUAL_WIELD) & go
    dw_hand = jnp.where(dw, jnp.minimum(copies, room), 0)
    dw_disc = jnp.where(dw, copies, 0) - dw_hand
    p = p.at[api.PILE_HAND].add(-from_hand + from_exh + one * dw_hand)
    p = p.at[api.PILE_DISCARD].add(-from_disc + one * dw_disc)
    p = p.at[api.PILE_EXHAUST].add(-from_exh + to_exh)
    p = p.at[api.PILE_DRAW].add(to_top)
    s = s._replace(piles=p.astype(I16))
    s = push_top(s, d, go & (to_top.sum() > 0))
    ex = to_exh.sum()
    s = cards.returns_on_exhaust(s, to_exh, go)
    s, jugg, trig = hooks.on_exhaust(s, ex, jnp.sum(to_exh * cards.rows(s)[:, tc.col("exhaust_energy")]), go & (ex > 0))
    return s, jugg, trig


def choice_point(spec: api.EnvSpec, s: State, kind, limbo, jugg, trig):
    """Open a choice: nothing (0 valid copies), resolve it (1), or leave it pending (2+).
    Returns (state, pending, Juggernaut events, triggered draws)."""
    v = valid(s, kind)
    n = v.sum()
    active = (kind > 0) & ~s.done
    auto = active & (n == 1)
    pending = active & (n >= 2)
    s, j2, t2 = apply_pick(spec, s, kind, jnp.argmax(v > 0), limbo, auto)
    choice = jnp.where(pending, jnp.stack([kind, limbo, jugg + j2, trig + t2]).astype(I16), s.choice)
    return s._replace(choice=choice), pending, jugg + j2, trig + t2


def pick_mask(s: State):
    """[D] legal picks while a choice is pending."""
    return valid(s, s.choice[KIND].astype(I32)) > 0
