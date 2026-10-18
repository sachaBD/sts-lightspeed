"""init: StartSpec -> State. Validates on device (errors become ERR_INVALID_START, never exceptions)."""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, U32, State


def init(spec: api.EnvSpec, start: api.StartSpec, key) -> State:
    del key  # used once combat start draws / rolls exist
    z = lambda *shape: jnp.zeros(shape, I16)
    count = start.count.astype(I16)
    valid = (
        jnp.all(count >= 0)
        & jnp.all((start.card >= 0) & (start.card < spec.n_cards))
        & jnp.all((start.card == 0) == (count == 0))
        & (start.hp > 0) & (start.hp <= start.max_hp)
        & jnp.all(start.potion_count >= 0)
        & jnp.all((start.potions == 0) == (start.potion_count == 0))
        & (start.potion_count.sum() <= start.potion_capacity) & (start.potion_capacity <= spec.P)
    )
    err = jnp.where(valid, U32(0), U32(api.ERR_INVALID_START))
    return State(
        slot_card=start.card.astype(I16), slot_up=start.upgraded.astype(I16),
        slot_cost=z(spec.D), slot_misc=z(spec.D),  # TODO(cards): cost from table
        piles=z(api.NUM_PILES, spec.D).at[api.PILE_DRAW].set(count),
        known_top=jnp.full((spec.K_TOP,), -1, I16),
        hp=start.hp.astype(I16), max_hp=start.max_hp.astype(I16), block=I16(0), energy=I16(0),
        p_status=z(spec.n_player_status),
        relics=start.relics.astype(I16), relic_ctr=z(spec.R), potions=start.potions.astype(I16),
        potion_n=start.potion_count.astype(I16), potion_cap=start.potion_capacity.astype(I16),
        e_hp=I16(420), e_max_hp=I16(420), e_block=I16(0),  # TODO(champ): hp roll + ascension
        e_move=I16(0), e_last=I16(0), e_prev=I16(0), e_phase2=jnp.bool_(False), e_stances=I16(0),
        e_status=z(spec.n_monster_status),
        choice=z(spec.CHOICE), turn=I16(0), ascension=start.ascension.astype(I16),
        done=~valid, won=jnp.bool_(False), truncated=~valid, error=err,
    )
