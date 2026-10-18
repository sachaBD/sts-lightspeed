# jaxsim

A pure-function JAX simulator for Ironclad vs The Champ. Design: [DESIGN.md](DESIGN.md). Contract:
[src/jaxsim/api.py](src/jaxsim/api.py).

**Status:** skeleton. The contract, ids, StartSpec builder and tests are in place. The game logic is a stub:
END_TURN advances the turn until the truncation cap.

```
uv sync                         # Python 3.12 + jax (CPU)
uv sync --extra cuda            # NVIDIA GPU
uv run pytest
uv run python bench/bench_step.py 4096
uv run python tools/gen_ids.py  # after changing C++ enums (bumps the contract)
```

Using it from another project:

```python
import jax, jaxsim

env = jaxsim.make_env()                                   # static; close over it in your jitted code
start = jaxsim.make_start(["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"], hp=80)
s = env.init(start, jax.random.key(0))
s = env.step(s, env.spec.END_TURN, jax.random.key(1))
obs, out, mask = env.observe(s), env.outcome(s), env.legal_mask(s)
```

Install into sts-ml with `uv pip install -e ../sts-lightspeed/jaxsim` (or a path dependency).
