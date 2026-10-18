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
        slot_cost=jnp.where(count > 0, rows[:, tc.col("cost")], 0).astype(I16), slot_misc=z(spec.D),
        piles=z(api.NUM_PILES, spec.D).at[api.PILE_DRAW].set(count.astype(I16)),
        known_top=jnp.full((spec.K_TOP,), -1, I16),
        hp=start.hp.astype(I16), max_hp=start.max_hp.astype(I16), block=I16(0), energy=I16(3),
        p_str=I16(0), p_dex=I16(0), p_vuln=I16(0), p_weak=I16(0), p_frail=I16(0), p_ja=I16(0),
        relics=start.relics.astype(I16), relic_ctr=z(spec.R), potions=start.potions.astype(I16),
        potion_n=start.potion_count.astype(I16), potion_cap=start.potion_capacity.astype(I16),
        e_hp=I16(champ.HP), e_max_hp=I16(champ.HP), e_block=I16(0), e_str=I16(0), e_vuln=I16(0), e_weak=I16(0),
        e_metallicize=I16(0), e_move=I16(0), e_last=I16(0), e_ai=I16(0),
        choice=z(spec.CHOICE), turn=I16(0),
        done=bad, won=jnp.bool_(False), truncated=bad, error=err,
    )
    k_draw, k_roll = jax.random.split(key)
    roll = jax.random.randint(k_roll, (), 0, 100) if forced is None else forced.roll
    s = champ.roll(s, roll)
    s, _ = piles.draw(spec, s, jnp.where(bad, 0, 5), k_draw, forced, 5)
    return s
