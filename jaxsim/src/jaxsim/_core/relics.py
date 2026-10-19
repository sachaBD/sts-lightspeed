"""Relic effects at combat start and turn start (BattleContext::initRelics, Player::applyStartOfTurnRelics).

Relics are flags in State.rel (one per relic id); `has(s, "NAME")` is a static-index lookup. Counters that carry
over from the run come from StartSpec.relic_data.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import hooks
from jaxsim._core.state import I16, I32, State
from jaxsim._core.tables import relics as tr

i16 = hooks.i16


def has(s: State, name: str):
    return s.rel[tr.ID[name]]


def data(start: api.StartSpec, name: str):
    """The run's data for a relic (0 if not held)."""
    return jnp.sum(jnp.where(start.relics == tr.ID[name], start.relic_data, 0)).astype(I32)


def debuff_player(s: State, field: str, bit: int, amount, monster_source=True) -> State:
    """Player::debuff: Artifact negates it (and is used up); a monster-sourced debuff is just-applied only if new."""
    from jaxsim._core import champ
    amount = jnp.asarray(amount, I32)
    blocked = (amount != 0) & (s.p_artifact > 0)
    s = s._replace(p_artifact=i16(s.p_artifact - blocked))
    amount = jnp.where(blocked, 0, amount)
    if bit:
        cur = getattr(s, field)
        new = (amount != 0) & (cur == 0) & monster_source
        return s._replace(**{field: i16(cur + amount)}, p_ja=i16(s.p_ja | jnp.where(new, bit, 0)))
    return s._replace(**{field: i16(getattr(s, field) + amount)})


def combat_start_pre_draw(spec: api.EnvSpec, s: State, start: api.StartSpec) -> State:
    """initRelics, before the opening draw."""
    h = lambda n: has(s, n)
    p_str = s.p_str + h("VAJRA") + data(start, "GIRYA") + data(start, "DU_VU_DOLL") + 3 * h("MUTAGENIC_STRENGTH") \
        + 2 * h("BRIMSTONE")
    s = s._replace(p_str=i16(p_str), p_dex=i16(s.p_dex + h("ODDLY_SMOOTH_STONE")),
                   e_str=i16(s.e_str + h("BRIMSTONE") + h("PHILOSOPHERS_STONE")))
    s = debuff_player(s, "p_lose_str", 0, jnp.where(h("MUTAGENIC_STRENGTH"), 3, 0))
    energy = sum(h(n).astype(I32) for n in tr.ENERGY)
    heal = 2 * h("BLOOD_VIAL") + 25 * h("PANTOGRAPH")
    flower = data(start, "HAPPY_FLOWER") + 1
    flower_now = h("HAPPY_FLOWER") & (flower == 3)
    incense = data(start, "INCENSE_BURNER") + 1
    incense_now = h("INCENSE_BURNER") & (incense == 6)
    return s._replace(
        p_energy_per_turn=i16(s.p_energy_per_turn + energy), block=i16(s.block + 10 * h("ANCHOR")),
        hp=i16(jnp.minimum(s.hp + heal, s.max_hp)),
        energy=i16(s.energy + h("LANTERN") + flower_now),
        p_thorns=i16(s.p_thorns + 3 * h("BRONZE_SCALES")), p_plated=i16(s.p_plated + 4 * h("THREAD_AND_NEEDLE")),
        p_buffer=i16(s.p_buffer + h("FOSSILIZED_HELIX")),
        r_happy_flower=i16(jnp.where(flower_now, 0, jnp.where(h("HAPPY_FLOWER"), flower, 0))),
        r_incense=i16(jnp.where(incense_now, 0, jnp.where(h("INCENSE_BURNER"), incense, 0))),
        p_intangible=i16(s.p_intangible + incense_now))


def combat_start_post_draw(spec: api.EnvSpec, s: State) -> State:
    """atBattleStart relics after the opening draw (Bag of Preparation's draw is done by init), then Mercury
    Hourglass."""
    from jaxsim._core import champ, piles
    from jaxsim._core.tables import cards as tc
    h = lambda n: has(s, n)
    # Bag of Marbles: DebuffAllEnemy(.., false); Red Mask: DebuffAllEnemy default (monster-sourced: just applied)
    red = h("RED_MASK") & (s.e_weak == 0)
    s = s._replace(e_vuln=i16(s.e_vuln + h("BAG_OF_MARBLES")), e_weak=i16(s.e_weak + h("RED_MASK")),
                   e_ja=i16(s.e_ja | jnp.where(red, 2, 0)))
    # Gremlin Visage's Weak applies immediately; Clockwork Souvenir's Artifact is a queued action (later)
    s = debuff_player(s, "p_weak", champ.JA_WEAK, jnp.where(h("GREMLIN_VISAGE"), 1, 0))
    s = s._replace(p_artifact=i16(s.p_artifact + h("CLOCKWORK_SOUVENIR")))
    return hooks.hurt_enemy(s, 3 * h("MERCURY_HOURGLASS").astype(I32), ~s.done & h("MERCURY_HOURGLASS"))


def mark_of_pain(s: State, want) -> State:
    """Mark of Pain: 2 Wounds shuffled into the draw pile at combat start."""
    from jaxsim._core import piles
    from jaxsim._core.tables import cards as tc
    wound = tc._ID["WOUND"]
    s, slot, ok = piles.find_or_alloc(s, wound, 0, int(tc.TABLE[wound, 0, tc.col("cost")]), 0, want)
    return s._replace(piles=s.piles.at[api.PILE_DRAW, slot].add(jnp.where(want & ok, 2, 0).astype(I16)))


def turn_start(s: State) -> State:
    """applyStartOfTurnRelics (after the Champ's turn and the block reset). s.turn is already the new turn index."""
    go = ~s.done
    h = lambda n: has(s, n) & go
    s = s._replace(p_str=i16(s.p_str + 2 * h("BRIMSTONE")), e_str=i16(s.e_str + h("BRIMSTONE")))
    flower = s.r_happy_flower + h("HAPPY_FLOWER")
    flower_now = flower == 3
    incense = s.r_incense + h("INCENSE_BURNER")
    incense_now = incense == 6
    s = s._replace(r_happy_flower=i16(jnp.where(flower_now, 0, flower)), r_incense=i16(jnp.where(incense_now, 0, incense)),
                   p_intangible=i16(s.p_intangible + incense_now))
    # queued actions (after the energy recharge and block reset): energy, block, damage
    energy_bonus = (h("ART_OF_WAR") & (s.p_attacks_played == 0)).astype(I32) + flower_now
    s, ev1 = hooks.gain_block(s, jnp.where(h("HORN_CLEAT") & (s.turn == 1), 14, 0), go)
    s, ev2 = hooks.gain_block(s, jnp.where(h("CAPTAINS_WHEEL") & (s.turn == 2), 18, 0), go)
    s = hooks.juggernaut(s, ev1 + ev2)
    s = hooks.hurt_enemy(s, 3 * h("MERCURY_HOURGLASS").astype(I32), ~s.done & h("MERCURY_HOURGLASS"))
    return s, energy_bonus
