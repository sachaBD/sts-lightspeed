"""Randomness. Public step/init sample from the key. The parity harness instead passes `Forced` outcomes taken
from the C++ (which cards were drawn, The Champ's roll) through the same code paths. `forced=None` is a Python
value, so the production path has no forced-mode cost.
"""
from typing import NamedTuple

import jax


class Forced(NamedTuple):
    draw: jax.Array   # [D] int32: multiset of slots this call draws, consumed lowest slot first
    roll: jax.Array   # [] int32: aiRng.random(99) value for The Champ's roll in this call
