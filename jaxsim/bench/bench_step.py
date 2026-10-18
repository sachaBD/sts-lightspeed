"""Random-policy throughput: vmap over games, scan over decisions, auto-reset. Usage: python bench/bench_step.py [batch ...]"""
import sys
import time

import jax
import jax.numpy as jnp

import jaxsim

STEPS = 256
DECK = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"]


def make_runner(env, batch):
    start = jaxsim.stack_starts([jaxsim.make_start(DECK, hp=80)] * batch)
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    legal, outcome = jax.vmap(env.legal_mask), jax.vmap(env.outcome)

    def one(carry, _):
        s, key, fights = carry
        key, ka, ks, kr = jax.random.split(key, 4)
        a = jax.random.categorical(ka, jnp.where(legal(s), 0.0, -jnp.inf))
        s = step(s, a, jax.random.split(ks, batch))
        done = outcome(s).done
        fresh = init(start, jax.random.split(kr, batch))
        s = jax.tree.map(lambda f, x: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x), fresh, s)
        return (s, key, fights + done.sum()), None

    @jax.jit
    def run(key):
        s = init(start, jax.random.split(key, batch))
        (_, _, fights), _ = jax.lax.scan(one, (s, key, jnp.int32(0)), None, length=STEPS)
        return fights

    return run


def main():
    env = jaxsim.make_env()
    print(jax.devices(), jaxsim.CONTRACT)
    for b in [int(x) for x in sys.argv[1:]] or [256, 1024, 4096]:
        run = make_runner(env, b)
        t0 = time.perf_counter(); jax.block_until_ready(run(jax.random.key(0))); compile_s = time.perf_counter() - t0
        t0 = time.perf_counter(); fights = int(jax.block_until_ready(run(jax.random.key(1)))); dt = time.perf_counter() - t0
        print(f"batch {b:6d}: {b * STEPS / dt:12,.0f} decisions/s  {fights / dt:10,.0f} fights/s  (compile {compile_s:.1f}s)")


if __name__ == "__main__":
    main()
