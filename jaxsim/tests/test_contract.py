"""Contract tests: everything api.py promises. Must hold for every implementation stage."""
import jax
import jax.numpy as jnp
import numpy as np

import jaxsim
from jaxsim import api


def trees_equal(a, b):
    return all(jax.tree.leaves(jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)))


def random_rollout(env, s, key, n):
    """Random legal actions; returns the list of states."""
    out = [s]
    for _ in range(n):
        key, ka, ks = jax.random.split(key, 3)
        a = jax.random.categorical(ka, jnp.where(env.legal_mask(s), 0.0, -jnp.inf))
        s = env.step(s, a, ks)
        out.append(s)
    return out


def test_contract_version():
    assert jaxsim.CONTRACT == "jaxsim_champ_v1.0"


def test_action_layout(env):
    sp = env.spec
    assert sp.END_TURN == sp.D and sp.potion_action(0) == sp.D + 1 and sp.DONE == sp.D + 1 + sp.P
    assert sp.A == sp.DONE + 1


def test_shapes_and_dtypes(env, start, key):
    sp = env.spec
    s = env.init(start, key)
    o, out, m = env.observe(s), env.outcome(s), env.legal_mask(s)
    assert m.shape == (sp.A,) and m.dtype == jnp.bool_
    expect = dict(slot_card=(sp.D,), slot_upgraded=(sp.D,), slot_cost=(sp.D,), slot_misc=(sp.D,),
                  piles=(api.NUM_PILES, sp.D), known_top=(sp.K_TOP,), player=(5,),
                  player_status=(sp.n_player_status,), enemy=(6,), enemy_status=(sp.n_monster_status,),
                  relics=(sp.R,), relic_counters=(sp.R,), potions=(sp.P,), potion_counts=(sp.P,), potion_capacity=(), choice=(sp.CHOICE,), turn=())
    assert set(expect) == set(o._fields)
    for f, shape in expect.items():
        x = getattr(o, f)
        assert x.shape == shape, f
        assert x.dtype == jnp.int16, f
    for f in ("done", "won", "truncated"):
        assert getattr(out, f).dtype == jnp.bool_
    assert out.error.dtype == jnp.uint32 and out.hp.dtype == jnp.int16


def test_init_valid(env, start, key):
    s = env.init(start, key)
    out = env.outcome(s)
    assert not out.done and int(out.error) == 0
    assert int(env.observe(s).piles.sum()) == int(start.count.sum())


def test_invalid_start_flags_error(env, start, key):
    bads = [
        start._replace(count=start.count.at[-1].set(3)),                     # count on an empty card slot
        start._replace(potion_count=start.potion_count.at[-1].set(1)),       # count on an empty potion slot
        start._replace(potions=start.potions.at[0].set(1), potion_count=start.potion_count.at[0].set(3)),  # > capacity
    ]
    for bad in bads:
        out = env.outcome(env.init(bad, key))
        assert out.done and out.truncated and int(out.error) & api.ERR_INVALID_START


def test_deterministic(env, start, key):
    s = env.init(start, key)
    a = jnp.int32(env.spec.END_TURN)
    assert trees_equal(env.step(s, a, key), env.step(s, a, key))


def test_jit_matches_eager(env, start, key):
    s0 = env.init(start, key)
    assert trees_equal(jax.jit(env.init)(start, key), s0)
    a = jnp.int32(env.spec.END_TURN)
    assert trees_equal(jax.jit(env.step)(s0, a, key), env.step(s0, a, key))
    assert trees_equal(jax.jit(env.observe)(s0), env.observe(s0))
    assert trees_equal(jax.jit(env.legal_mask)(s0), env.legal_mask(s0))


def test_vmap_matches_loop(env, start, key):
    n = 4
    starts = jaxsim.stack_starts([start] * n)
    keys = jax.random.split(key, n)
    sb = jax.vmap(env.init)(starts, keys)
    actions = jnp.full((n,), env.spec.END_TURN, jnp.int32)
    nb = jax.vmap(env.step)(sb, actions, keys)
    for i in range(n):
        si = env.init(start, keys[i])
        assert trees_equal(jax.tree.map(lambda x: x[i], nb), env.step(si, actions[i], keys[i]))


def test_illegal_action_flags_error(env, start, key):
    s = env.init(start, key)
    for a in (-1, env.spec.A, int(np.flatnonzero(~np.asarray(env.legal_mask(s)))[0])):
        out = env.outcome(env.step(s, jnp.int32(a), key))
        assert out.done and out.truncated and int(out.error) & api.ERR_ILLEGAL_ACTION


def test_terminal_is_absorbing(env, start, key):
    s = env.init(start, key)
    for st in random_rollout(env, s, key, env.spec.max_turns + 5):
        if env.outcome(st).done:
            break
    assert env.outcome(st).done
    m = env.legal_mask(st)
    assert bool(m[env.spec.END_TURN]) and int(m.sum()) == 1
    assert trees_equal(env.step(st, jnp.int32(env.spec.END_TURN), key), st)


def test_unsupported_flags_error(env, key):
    s = jaxsim.make_start(["BASH"], hp=10, potions=["ATTACK_POTION"], allow_unsupported=True)
    out = env.outcome(env.init(s, key))
    assert out.done and out.truncated and int(out.error) & api.ERR_UNSUPPORTED


def test_truncation_is_not_a_loss(start, key):
    env = jaxsim.make_env(jaxsim.EnvSpec(max_turns=2))
    s = env.init(start, key)
    states = [s]
    for _ in range(3):
        states.append(env.step(states[-1], jnp.int32(env.spec.END_TURN), key))
    out = env.outcome(states[-1])
    assert out.done and out.truncated and not out.won and int(out.error) == 0


def test_relic_counters_observed(env, key):
    start = jaxsim.make_start(["STRIKE_RED"] * 10, hp=80, relics=[("ANCHOR", 0), ("PEN_NIB", 5)])
    o = env.observe(env.init(start, key))
    assert np.asarray(o.relic_counters)[:2].tolist() == [0, 5]
