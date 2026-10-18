"""step and legal_mask.

One step = one decision. Both decision kinds are computed and selected (branch-free under vmap), around one
shared draw:
    play card:  play_pre -> draw(card draw) -> play_post
    END_TURN:   end_turn_pre (discard hand, Champ turn, end of round, next turn start) -> draw(5) -> energy
"""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import cards, champ, piles
from jaxsim._core.state import I16, I32, U32, State, tree_where
from jaxsim._core.tables import cards as tc

HAND_DRAW = 5
MAX_DRAW = max(HAND_DRAW, int(tc.TABLE[:, :, tc.col("draw")].max()))


def legal_mask(spec: api.EnvSpec, s: State):
    mask = jnp.zeros((spec.A,), bool)
    mask = mask.at[: spec.D].set(cards.playable(s) & ~s.done)
    return mask.at[spec.END_TURN].set(True)


def end_turn_pre(spec: api.EnvSpec, s: State, roll) -> State:
    # discard the hand; ethereal cards exhaust
    p = s.piles.astype(I32)
    eth = cards.rows(s)[:, tc.col("ethereal")]
    hand = p[api.PILE_HAND]
    p = (p.at[api.PILE_EXHAUST].add(hand * eth).at[api.PILE_DISCARD].add(hand * (1 - eth))
          .at[api.PILE_HAND].set(0))
    s = s._replace(piles=p.astype(I16))
    s = champ.take_turn(s, roll)
    nxt = champ.end_of_round(s)
    turn = nxt.turn + 1
    cap = turn >= spec.max_turns
    nxt = nxt._replace(turn=turn.astype(I16), block=I16(0), done=nxt.done | cap, truncated=nxt.truncated | cap)
    return tree_where(s.done, s, nxt)  # died during the Champ's turn: frozen at that moment


def step(spec: api.EnvSpec, s: State, action, key, forced=None) -> State:
    action = jnp.asarray(action, I32)
    a = jnp.clip(action, 0, spec.A - 1)
    ok = (action >= 0) & (action < spec.A) & legal_mask(spec, s)[a]
    is_end = a == spec.END_TURN
    d = jnp.clip(a, 0, spec.D - 1)
    k_draw, k_roll = jax.random.split(key)
    roll = jax.random.randint(k_roll, (), 0, 100) if forced is None else forced.roll

    played, n_play = cards.play_pre(spec, s, d)
    ended = end_turn_pre(spec, s, roll)
    mid = tree_where(is_end, ended, played)
    n = jnp.where(mid.done, 0, jnp.where(is_end, HAND_DRAW, n_play))
    mid, _ = piles.draw(spec, mid, n, k_draw, forced, MAX_DRAW)
    ended = mid._replace(energy=jnp.where(mid.hp > 0, I16(3), mid.energy))
    nxt = tree_where(is_end, ended, cards.play_post(spec, mid, d))

    # one full select: illegal actions and terminal states keep s; then patch flags
    take = ok & ~s.done
    out = tree_where(take, nxt, s)
    bad = ~ok & ~s.done
    return out._replace(done=out.done | bad, truncated=out.truncated | bad,
                        error=out.error | jnp.where(bad, U32(api.ERR_ILLEGAL_ACTION), U32(0)))
