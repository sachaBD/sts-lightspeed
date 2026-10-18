"""Internal game state: one fixed-shape pytree per game. Private; free to change without a CONTRACT bump.

Layout rules: hot values are separate scalars, not entries of shared arrays (PR #2 measurements). Arithmetic
is done in int32 (float32 for damage, as the C++ does) and stored as int16.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

I16 = jnp.int16
I32 = jnp.int32
U32 = jnp.uint32


class State(NamedTuple):
    # card slots: identity per slot, counts per pile
    slot_card: jax.Array      # [D]
    slot_up: jax.Array        # [D]
    slot_cost: jax.Array      # [D] identity cost (-2 unplayable, -1 X)
    slot_misc: jax.Array      # [D]
    piles: jax.Array          # [NUM_PILES, D]
    known_top: jax.Array      # [K_TOP], -1 = none
    # player
    hp: jax.Array
    max_hp: jax.Array
    block: jax.Array
    energy: jax.Array
    p_str: jax.Array
    p_dex: jax.Array
    p_vuln: jax.Array
    p_weak: jax.Array
    p_frail: jax.Array
    p_ja: jax.Array           # just-applied bits for monster-sourced debuffs: 1 vuln, 2 weak, 4 frail
    p_lose_str: jax.Array     # Flex: strength lost at end of turn
    p_no_draw: jax.Array      # Battle Trance: no more draws this turn (stacks, like the C++ status)
    relics: jax.Array         # [R] ids (observation only for now)
    relic_ctr: jax.Array      # [R]
    potions: jax.Array        # [P] ids
    potion_n: jax.Array       # [P] counts
    potion_cap: jax.Array
    # enemy (The Champ)
    e_hp: jax.Array
    e_max_hp: jax.Array
    e_block: jax.Array
    e_str: jax.Array
    e_vuln: jax.Array
    e_weak: jax.Array
    e_metallicize: jax.Array
    e_move: jax.Array         # intent: next move (jaxsim move id)
    e_last: jax.Array         # move before the intent
    e_ai: jax.Array           # C++ miscInfo layout: bits 0-1 defensive stances used, bit 2 phase 2
    # flow
    choice: jax.Array         # [CHOICE]
    turn: jax.Array
    done: jax.Array           # bool
    won: jax.Array            # bool
    truncated: jax.Array      # bool
    error: jax.Array          # uint32


def tree_where(cond, a, b):
    """Per-leaf select between two states of identical structure."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)
