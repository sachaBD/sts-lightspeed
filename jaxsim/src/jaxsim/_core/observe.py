"""observe and outcome: the only views of State that consumers see."""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import State


def observe(spec: api.EnvSpec, s: State) -> api.Obs:
    return api.Obs(
        slot_card=s.slot_card, slot_upgraded=s.slot_up, slot_cost=s.slot_cost, slot_misc=s.slot_misc,
        piles=s.piles, known_top=s.known_top,
        player=jnp.stack([s.hp, s.max_hp, s.block, s.energy]), player_status=s.p_status,
        enemy=jnp.stack([s.e_hp, s.e_max_hp, s.e_block, s.e_move]), enemy_status=s.e_status,
        relics=s.relics, relic_counters=s.relic_ctr, potions=s.potions, choice=s.choice, turn=s.turn,
    )


def outcome(spec: api.EnvSpec, s: State) -> api.Outcome:
    return api.Outcome(done=s.done, won=s.won, truncated=s.truncated, hp=s.hp, max_hp=s.max_hp,
                       turn=s.turn, error=s.error)
