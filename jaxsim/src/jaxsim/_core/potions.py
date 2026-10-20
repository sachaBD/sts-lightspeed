"""Potions (BattleContext::drinkPotion): potion slots are identities with counts; action D+1+p drinks one of slot p.
Effects are the table columns, doubled by Sacred Bark; Toy Ornithopter heals 5 per potion. Fairy in a Bottle is
passive (hooks.lose_hp)."""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import hooks
from jaxsim._core.state import I16, I32, State
from jaxsim._core.tables import potions as tp
from jaxsim._core.tables import relics as tr

TABLE = jnp.asarray(tp.TABLE)
C = tp.col
i16 = hooks.i16


def usable(s: State):
    """[P] potion slots that can be drunk now."""
    ids = jnp.clip(s.potions.astype(I32), 0, TABLE.shape[0] - 1)
    return (s.potion_n > 0) & (TABLE[ids, C("drinkable")] == 1) & ~s.done


def drink(spec: api.EnvSpec, s: State, p):
    """Drink one potion of slot p. Returns (state, cards to draw, Juggernaut events)."""
    from jaxsim._core import champ, choice, relics
    r = TABLE[jnp.clip(s.potions[p].astype(I32), 0, TABLE.shape[0] - 1)]
    bark = jnp.where(s.rel[tr.ID["SACRED_BARK"]], 2, 1)
    v = lambda c: r[C(c)] * bark
    s = s._replace(potion_n=s.potion_n.at[p].add(-1).astype(I16))
    s = s._replace(potions=jnp.where(s.potion_n > 0, s.potions, I16(0)))
    s = hooks.heal(s, 5, s.rel[tr.ID["TOY_ORNITHOPTER"]])
    go = ~s.done
    s, jugg = hooks.gain_block(s, v("block"), go)
    s = hooks.heal(s, (s.max_hp.astype(I32) * v("heal_pct")) // 100, go & (r[C("heal_pct")] > 0))
    s = hooks.hurt_enemy(s, v("dmg_all") + v("dmg_target"), go & (r[C("dmg_all")] + r[C("dmg_target")] > 0))
    live = ~s.done
    belt = (r[C("e_vuln")] > 0) & live & s.rel[tr.ID["CHAMPION_BELT"]]
    s = s._replace(e_vuln=i16(s.e_vuln + v("e_vuln") * live), e_weak=i16(s.e_weak + v("e_weak") * live + belt),
                   p_str=i16(s.p_str + v("str") * live), p_dex=i16(s.p_dex + v("dex") * live),
                   energy=i16(s.energy + v("energy") * live), p_artifact=i16(s.p_artifact + v("artifact") * live),
                   p_plated=i16(s.p_plated + v("plated") * live),
                   p_metallicize=i16(s.p_metallicize + v("metallicize") * live),
                   p_thorns=i16(s.p_thorns + v("thorns") * live), p_intangible=i16(s.p_intangible + v("intangible") * live),
                   p_regen=i16(s.p_regen + v("regen") * live))
    s = relics.debuff_player(s, "p_lose_str", 0, v("lose_str") * live)
    s = relics.debuff_player(s, "p_lose_dex", 0, v("lose_dex") * live)
    # Fruit Juice: max hp (Player::increaseMaxHp heals the same amount)
    s = s._replace(max_hp=i16(s.max_hp + v("max_hp") * live))
    s = hooks.heal(s, v("max_hp"), live & (r[C("max_hp")] > 0))
    # Blessing of the Forge: upgrade the hand
    s = choice.upgrade_copies(s, jnp.where(live & (r[C("upgrade_hand")] > 0) & choice.can_upgrade(s),
                                           s.piles[api.PILE_HAND].astype(I32), 0))
    return s, v("draw") * live, jugg
