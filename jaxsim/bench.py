"""Random-policy throughput benchmark for the Champ slice.

Usage: python bench.py [batch ...]   (defaults to a sweep)
Reports agent decisions/s and fights/s with auto-reset, plus how many VM
iterations each step used (mean lane vs. the batch max that vmap pays for).
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import engine as E
import tables as T

STEPS = 256


def starter_deck():
    cards = ([T.STRIKE] * 4 + [T.DEFEND] * 4 + [T.BASH, T.POMMEL_STRIKE, T.SHRUG_IT_OFF, T.INFLAME,
             T.TWIN_STRIKE, T.IRON_WAVE, T.ANGER, T.WILD_STRIKE])
    cid = np.zeros(E.N, np.int32)
    up = np.zeros(E.N, np.int32)
    cid[:len(cards)] = cards
    up[len(cards) // 2:len(cards)] = 1
    return jnp.asarray(cid), jnp.asarray(up)


def make_runner(batch):
    cid, up = starter_deck()
    init = jax.vmap(lambda k: E.init_state(k, cid, up))
    step = jax.vmap(E.step)
    mask_fn = jax.vmap(E.legal_mask)

    def one(carry, _):
        s, key, fights, wins = carry
        key, k_act, k_reset = jax.random.split(key, 3)
        mask = mask_fn(s)
        action = jax.random.categorical(k_act, jnp.where(mask, 0.0, -1e9))
        s = step(s, action)
        fresh = init(jax.random.split(k_reset, batch))
        fights = fights + jnp.sum(s.done)
        wins = wins + jnp.sum(s.win)
        iters = s.vm_iters
        s = jax.tree.map(lambda f, x: jnp.where(
            s.done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x), fresh, s)
        return (s, key, fights, wins), (jnp.mean(iters), jnp.max(iters))

    @jax.jit
    def run(key):
        s = init(jax.random.split(key, batch))
        (s, _, fights, wins), (mean_it, max_it) = jax.lax.scan(
            one, (s, key, jnp.int32(0), jnp.int32(0)), None, length=STEPS)
        return fights, wins, jnp.mean(mean_it), jnp.mean(max_it)

    return run


def main():
    import os
    extra = int(os.environ.get("EXTRA_OPS", "0"))
    if extra:
        E.BRANCHES = E.padded_branches(extra, os.environ.get("PAD_KIND") == "scalar")
    print(jax.devices(), "opcodes:", len(E.BRANCHES))
    batches = [int(x) for x in sys.argv[1:]] or [256, 1024, 4096, 16384]
    for b in batches:
        run = make_runner(b)
        t0 = time.perf_counter()
        jax.block_until_ready(run(jax.random.PRNGKey(0)))
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fights, wins, mean_it, max_it = jax.block_until_ready(run(jax.random.PRNGKey(1)))
        dt = time.perf_counter() - t0
        print(f"batch {b:6d}: {b * STEPS / dt:12,.0f} decisions/s  {int(fights) / dt:10,.0f} fights/s  "
              f"win {int(wins) / max(int(fights), 1):.2f}  vm iters/step mean {float(mean_it):.1f} "
              f"max {float(max_it):.1f}  (compile {compile_s:.1f}s)")


if __name__ == "__main__":
    main()
