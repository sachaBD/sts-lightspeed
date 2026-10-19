"""init: StartSpec -> State at the first player decision. Validates on device (errors become ERR_* bits)."""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import champ, piles
from jaxsim._core.state import I16, I32, U32, State
from jaxsim._core.tables import cards as tc


def init(spec: api.EnvSpec, start: api.StartSpec, key, forced=None) -> State:
    z = lambda *shape: jnp.zeros(shape, I16)
    card = start.card.astype(I32)
    up = start.upgraded.astype(I32)
    count = start.count.astype(I32)
    in_range = (card >= 0) & (card < spec.n_cards)
    rows = jnp.asarray(tc.TABLE)[jnp.clip(card, 0, spec.n_cards - 1), jnp.minimum(up, 1)]
    valid = (
        jnp.all(count >= 0) & jnp.all(in_range) & jnp.all((card == 0) == (count == 0))
        & (start.hp > 0) & (start.hp <= start.max_hp)
        & jnp.all(start.potion_count >= 0)
        & jnp.all((start.potions == 0) == (start.potion_count == 0))
        & (start.potion_count.sum() <= start.potion_capacity) & (start.potion_capacity <= spec.P)
    )
    supported = jnp.all((count == 0) | (rows[:, tc.col("supported")] == 1)) & jnp.all(start.potions == 0)
    err = jnp.where(valid, U32(0), U32(api.ERR_INVALID_START)) | jnp.where(supported, U32(0), U32(api.ERR_UNSUPPORTED))
    bad = err != 0
    s = State(
        slot_card=card.astype(I16), slot_up=up.astype(I16),
        slot_cost=jnp.where(count > 0, rows[:, tc.col("cost")], 0).astype(I16), slot_misc=jnp.where(count > 0, start.misc, 0).astype(I16),
        piles=z(api.NUM_PILES, spec.D).at[api.PILE_DRAW].set(count.astype(I16)),
        known_top=jnp.full((spec.K_TOP,), -1, I16),
        hp=start.hp.astype(I16), max_hp=start.max_hp.astype(I16), block=I16(0), energy=I16(0),
        p_str=I16(0), p_dex=I16(0), p_vuln=I16(0), p_weak=I16(0), p_frail=I16(0), p_ja=I16(0), p_lose_str=I16(0), p_no_draw=I16(0), p_energy_per_turn=I16(3), p_intangible=I16(0), p_cards_played=I16(0), p_attacks_played=I16(0), p_artifact=I16(0), p_buffer=I16(0),
        p_thorns=I16(0), p_plated=I16(0), r_happy_flower=I16(0), r_incense=I16(0),
        rel=jnp.zeros(spec.n_relics, bool).at[jnp.clip(start.relics.astype(I32), 0, spec.n_relics - 1)].set(start.relics > 0),
        p_metallicize=I16(0), p_demon_form=I16(0), p_barricade=I16(0), p_brutality=I16(0), p_combust=I16(0), p_combust_hp=I16(0), p_corruption=I16(0), p_dark_embrace=I16(0), p_evolve=I16(0), p_fnp=I16(0), p_fire_breathing=I16(0), p_juggernaut=I16(0), p_rage=I16(0), p_rupture=I16(0), p_flame_barrier=I16(0),
        relics=start.relics.astype(I16), relic_ctr=z(spec.R), potions=start.potions.astype(I16),
        potion_n=start.potion_count.astype(I16), potion_cap=start.potion_capacity.astype(I16),
        e_hp=I16(champ.HP), e_max_hp=I16(champ.HP), e_block=I16(0), e_str=I16(0), e_vuln=I16(0), e_weak=I16(0),
        e_metallicize=I16(0), e_move=I16(0), e_last=I16(0), e_ai=I16(0), e_ja=I16(0),
        choice=z(spec.CHOICE), turn=I16(0),
        done=bad, won=jnp.bool_(False), truncated=bad, error=err,
    )
    k_draw, k_roll = jax.random.split(key)
    roll = jax.random.randint(k_roll, (), 0, 100) if forced is None else forced.roll
    s = champ.roll(s, roll)
    # innate cards start on top of the draw pile: all of them are drawn, then the rest of the 5
    innate = jnp.where(rows[:, tc.col("innate")] == 1, count, 0)
    innate = jnp.minimum(innate, jnp.maximum(spec.H - jnp.cumsum(innate) + innate, 0)).astype(I16)
    s = s._replace(piles=s.piles.at[api.PILE_DRAW].add(-innate).at[api.PILE_HAND].add(innate))
    from jaxsim._core import relics
    s = relics.combat_start_pre_draw(spec, s, start)
    s = s._replace(energy=(s.energy + s.p_energy_per_turn).astype(I16))
    k_draw, k_bag = jax.random.split(k_draw)
    s, forced, _, _ = piles.draw(spec, s, jnp.where(bad, 0, jnp.maximum(5 - innate.sum(), 0)), k_draw, forced, extend=False)
    # atBattleStart in relic order: Mark of Pain's Wounds before or after Bag of Preparation's 2 draws
    pos = lambda n: jnp.argmax(start.relics == relics.tr.ID[n])
    mop_first = pos("MARK_OF_PAIN") < pos("BAG_OF_PREPARATION")
    s = relics.mark_of_pain(s, relics.has(s, "MARK_OF_PAIN") & mop_first)
    s, _, _, _ = piles.draw(spec, s, jnp.where(relics.has(s, "BAG_OF_PREPARATION") & ~bad, 2, 0), k_bag, forced, extend=False)
    s = relics.mark_of_pain(s, relics.has(s, "MARK_OF_PAIN") & ~mop_first)
    return relics.combat_start_post_draw(spec, s)
