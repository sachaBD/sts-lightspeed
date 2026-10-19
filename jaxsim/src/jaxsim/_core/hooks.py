"""Shared event helpers: block gain, damage to the enemy, hp loss, exhaust triggers.

Every helper takes `go` (bool): the event only happens if the fight is still going when it would. After the
fight ends nothing changes the outcome (a dead player cannot win and vice versa).
"""
import jax.numpy as jnp

from jaxsim._core.state import I16, I32, State
from jaxsim._core.tables import cards as tc

i16 = lambda x: jnp.asarray(x).astype(I16)


def gain_block(s: State, amount, go):
    """Player::gainBlock: returns (state, Juggernaut events: 1 if block was gained)."""
    ev = (amount > 0) & go
    return s._replace(block=i16(s.block + jnp.where(ev, amount, 0))), ev.astype(I32)


def hurt_enemy(s: State, damage, go) -> State:
    """Monster::damage / attacked with already-calculated damage: block absorbs, then hp; may win."""
    damage = jnp.where(go, damage, 0)
    blocked = jnp.minimum(s.e_block.astype(I32), damage)
    e_hp = s.e_hp - (damage - blocked)
    won = go & (e_hp <= 0)
    return s._replace(e_block=i16(s.e_block - blocked), e_hp=i16(jnp.maximum(e_hp, 0)), won=s.won | won,
                      done=s.done | won)


def juggernaut(s: State, events) -> State:
    """Juggernaut deals its amount per block-gain event (each one a separate DamageRandomEnemy)."""
    return hurt_enemy(s, events * s.p_juggernaut.astype(I32), ~s.done & (s.p_juggernaut > 0))


def blood_events(s: State, events) -> State:
    """CardManager::onTookDamage on every hp-loss event: Blood for Blood copies in the draw pile, hand and discard
    cost 1 less (not below 0); exhausted copies keep their cost."""
    from jaxsim._core import cards, piles
    from jaxsim import api
    blood = (cards.rows(s)[:, tc.col("blood_cost")] == 1) & (s.slot_cost > 0) & (events > 0)
    move = jnp.zeros_like(s.piles, I32)
    for pile in (api.PILE_DRAW, api.PILE_HAND, api.PILE_DISCARD):
        move = move.at[pile].set(jnp.where(blood, s.piles[pile], 0).astype(I32))
    return piles.retarget(s, move, s.slot_card, s.slot_up, jnp.maximum(s.slot_cost - events, 0), s.slot_misc)


def _rel(s: State, name: str):
    from jaxsim._core.tables import relics as tr
    return s.rel[tr.ID[name]]


def heal(s: State, amount, go) -> State:
    """Player::heal: Mark of the Bloom blocks it, Magic Flower x1.5; Red Skull's Strength goes when healed above
    half."""
    from jaxsim._core import relics
    amount = jnp.where(_rel(s, "MAGIC_FLOWER"), amount * 3 // 2, amount)
    amount = jnp.where(go & ~_rel(s, "MARK_OF_THE_BLOOM"), amount, 0)
    was_low = s.hp <= s.max_hp // 2
    hp = jnp.minimum(s.hp + amount, s.max_hp)
    s = s._replace(hp=i16(hp))
    leaves_low = _rel(s, "RED_SKULL") & was_low & (hp > s.max_hp // 2) & (amount > 0)
    return relics.debuff_player(s, "p_str", 0, jnp.where(leaves_low, -3, 0))


def lose_hp(s: State, amount, self_damage, go, events=1) -> State:
    """Player::loseHp / hpWasLost for `amount` (> 0) already past block, as `events` separate losses.
    Intangible makes each loss 1, Tungsten Rod takes 1 off each. Per loss: Rupture (self damage), Blood for Blood,
    Self-Forming Clay, Runic Cube; the first loss of the combat: Centennial Puzzle; Red Skull when it drops to
    half; death unless Lizard Tail (heals to half once). Draws from Runic Cube / Centennial Puzzle are done at the
    next draw point."""
    amount = jnp.asarray(amount, I32)
    events = jnp.asarray(events, I32)
    amount = jnp.where((s.p_intangible > 0) & (amount > 0), jnp.minimum(amount, events), amount)
    rod = _rel(s, "TUNGSTEN_ROD")
    amount = jnp.where(rod, jnp.maximum(amount - events, 0), amount)
    lost = go & (amount > 0)
    events = jnp.where(lost, events, 0)
    was_low = s.hp <= s.max_hp // 2
    hp = s.hp - jnp.where(lost, amount, 0)
    died = lost & (hp <= 0)
    centennial = _rel(s, "CENTENNIAL_PUZZLE") & lost
    from jaxsim._core.tables import relics as tr
    s = s._replace(hp=i16(jnp.maximum(hp, 0)),
                   p_str=i16(s.p_str + jnp.where(self_damage, events * s.p_rupture, 0)
                             + 3 * (_rel(s, "RED_SKULL") & lost & ~was_low & (hp <= s.max_hp // 2))),
                   p_next_turn_block=i16(s.p_next_turn_block + 3 * events * _rel(s, "SELF_FORMING_CLAY")),
                   p_pending_draw=i16(s.p_pending_draw + events * _rel(s, "RUNIC_CUBE") + 3 * centennial),
                   rel=s.rel.at[tr.ID["CENTENNIAL_PUZZLE"]].set(_rel(s, "CENTENNIAL_PUZZLE") & ~lost))
    # Lizard Tail: instead of dying, heal to half max hp (once)
    lizard = died & _rel(s, "LIZARD_TAIL")
    s = s._replace(rel=s.rel.at[tr.ID["LIZARD_TAIL"]].set(_rel(s, "LIZARD_TAIL") & ~lizard))
    s = heal(s, s.max_hp.astype(I32) // 2, lizard)
    s = s._replace(done=s.done | (died & ~lizard))
    return blood_events(s, events)


def damage_player_hits(s: State, per, k, self_damage, go) -> State:
    """k separate Player::damage hits of `per` (Burns): block absorbs first; hit i loses hp iff i * per > block."""
    k = jnp.where(go, k, 0)
    per = jnp.where((s.p_intangible > 0) & (per > 0), 1, per)
    total = per * k
    blocked = jnp.minimum(s.block.astype(I32), total)
    events = jnp.where(per > 0, k - jnp.minimum(k, s.block.astype(I32) // jnp.maximum(per, 1)), 0)
    s = s._replace(block=i16(s.block - blocked))
    lost = total - blocked
    # Buffer negates the first hits that would lose hp (the first of them may be partly blocked)
    absorbed = jnp.minimum(s.p_buffer.astype(I32), events)
    first = lost - jnp.maximum(events - 1, 0) * per
    lost = lost - jnp.where(absorbed > 0, first + (absorbed - 1) * per, 0)
    s = s._replace(p_buffer=i16(s.p_buffer - absorbed))
    return lose_hp(s, lost, self_damage, go & (k > 0), events - absorbed)


def on_exhaust(s: State, n, sentinel_energy, go):
    """triggerAndMoveToExhaustPile for n cards: Sentinel energy (immediate), Feel No Pain block per card,
    Dark Embrace draws per card. Returns (state, Juggernaut events, cards to draw later)."""
    n = jnp.where(go, n, 0)
    s = s._replace(energy=i16(s.energy + jnp.where(go, sentinel_energy, 0)))
    s = hurt_enemy(s, 3 * n * _rel(s, "CHARONS_ASHES"), go & (n > 0) & _rel(s, "CHARONS_ASHES"))   # Charon's Ashes
    fnp = s.p_fnp.astype(I32)
    s = s._replace(block=i16(s.block + n * fnp))
    jugg = jnp.where(fnp > 0, n, 0)
    return s, jugg, n * s.p_dark_embrace.astype(I32)
