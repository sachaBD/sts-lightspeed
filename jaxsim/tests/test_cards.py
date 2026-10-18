"""Targeted card rules (parity covers them statistically; these pin known edge cases)."""
import jax
import jax.numpy as jnp
import numpy as np

import jaxsim
from jaxsim import api
from jaxsim._core.state import I16


def _with_hand(env, card, **fields):
    start = jaxsim.make_start([card, "STRIKE_RED"], hp=80)
    s = env.init(start, jax.random.key(0))
    slot = int(np.flatnonzero(np.asarray(s.slot_card) == jaxsim.CARD_NAMES.index(card))[0])
    piles = s.piles.at[api.PILE_HAND].set(0).at[api.PILE_HAND, slot].set(1)
    return s._replace(piles=piles, **{k: I16(v) for k, v in fields.items()}), slot


def test_iron_wave_block_applies_frail_and_dex_once():
    env = jaxsim.make_env()
    for fields, block in ((dict(p_frail=1), 3), (dict(p_dex=2), 7), (dict(p_dex=2, p_frail=1), 5), ({}, 5)):
        s, slot = _with_hand(env, "IRON_WAVE", **fields)
        s = env.step(s, jnp.int32(slot), jax.random.key(1))
        assert int(env.observe(s).player[2]) == block, fields
