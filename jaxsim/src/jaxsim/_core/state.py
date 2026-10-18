"""Internal game state: one fixed-shape pytree per game. Private; free to change without a CONTRACT bump.

Layout rule from the PR #2 benchmarks: hot values are separate scalars, not entries of a shared array.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

I16 = jnp.int16
U32 = jnp.uint32


class State(NamedTuple):
    # card slots: identity per slot, counts per pile
    slot_card: jax.Array      # [D]
    slot_up: jax.Array        # [D]
    slot_cost: jax.Array      # [D]
    slot_misc: jax.Array      # [D]
    piles: jax.Array          # [NUM_PILES, D]
    known_top: jax.Array      # [K_TOP], -1 = none
    # player
    hp: jax.Array
    max_hp: jax.Array
    block: jax.Array
    energy: jax.Array
    p_status: jax.Array       # [n_player_status] (hot statuses will move to scalars)
    relics: jax.Array         # [R]
    relic_ctr: jax.Array      # [R]
    potions: jax.Array        # [P]
    # enemy (The Champ)
    e_hp: jax.Array
    e_max_hp: jax.Array
    e_block: jax.Array
    e_move: jax.Array         # intent (move id)
    e_last: jax.Array
    e_prev: jax.Array
    e_phase2: jax.Array       # bool
    e_stances: jax.Array
    e_status: jax.Array       # [n_monster_status]
    # flow
    choice: jax.Array         # [CHOICE]
    turn: jax.Array
    ascension: jax.Array
    done: jax.Array           # bool
    won: jax.Array            # bool
    truncated: jax.Array      # bool
    error: jax.Array          # uint32


def tree_where(cond, a, b):
    """Per-leaf select between two states of identical structure."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)
