"""step and legal_mask. Currently the null game: END_TURN advances the turn until the truncation cap."""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, U32, State, tree_where


def legal_mask(spec: api.EnvSpec, s: State):
    mask = jnp.zeros((spec.A,), bool).at[spec.END_TURN].set(True)
    # TODO(cards/choice/potions): slot plays, picks, DONE, potions; terminal keeps END_TURN only.
    return mask


def _apply(spec: api.EnvSpec, s: State, a, key) -> State:
    del key
    end = a == spec.END_TURN
    turn = s.turn + end.astype(I16)
    cap = turn >= spec.max_turns
    return s._replace(turn=turn, done=s.done | cap, truncated=s.truncated | cap)


def _fail(s: State, bit: int) -> State:
    return s._replace(done=jnp.bool_(True), truncated=jnp.bool_(True), error=s.error | U32(bit))


def step(spec: api.EnvSpec, s: State, action, key) -> State:
    action = jnp.asarray(action, jnp.int32)
    in_range = (action >= 0) & (action < spec.A)
    a = jnp.clip(action, 0, spec.A - 1)
    ok = in_range & legal_mask(spec, s)[a]
    nxt = tree_where(ok, _apply(spec, s, a, key), _fail(s, api.ERR_ILLEGAL_ACTION))
    return tree_where(s.done, s, nxt)  # terminal states are absorbing
