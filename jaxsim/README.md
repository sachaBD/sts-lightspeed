# jaxsim

A very fast simulator of one Slay the Spire fight, **Ironclad vs The Champ**, written in JAX so that thousands
of fights run in parallel on a GPU. It is the environment for an AlphaZero-style training loop: the search,
the neural network and this simulator all run inside one compiled GPU program.

The C++ simulator in this repository is the reference. jaxsim is tested against it.

**Status:** the core game works and matches the C++ simulator exactly on 200 random fights. It supports A20
Champ fights with 12 cards (see `jaxsim.supported_cards()`). Relics and potions are not implemented yet.

## How it fits together

```
 sts-ml (the "play" side)                         jaxsim (the "sim" side, this package)
 ─────────────────────────                        ─────────────────────────────────────
 search (mctx), neural network,      uses ──►     the rules of the fight:
 rewards, replay buffer, training                 init, step, legal actions, observation, outcome
```

The two sides meet at one small interface (`jaxsim.Env`, defined in `src/jaxsim/api.py`). Because JAX compiles
both sides together, the boundary costs nothing at runtime.

## What the agent sees and does

A fight is described by **slots**. A card slot is one distinct card (for example "Strike", or "Pommel Strike+"),
with counts saying how many copies are in each pile. Identical copies share a slot, so the agent never has to
tell two Strikes apart. Potions work the same way. Slot order means nothing: the observation says which card is
in each slot, which is why one model can play any deck.

```
Observation
  cards    slot_card/upgraded/cost/misc [48]     piles [5 × 48]: draw hand discard exhaust choice
           known_top [2]
  potions  potions [5] ids   potion_counts [5]   potion_capacity
  player   hp max_hp block energy   player_status [87]
  enemy    hp max_hp block intent   enemy_status [43]
  relics   relics [32]  relic_counters [32]       choice [4]   turn

Actions (55)
  0..47 card slot d   48 END_TURN   49..53 potion slot p   54 DONE
```

- **Card slot d:** play one copy of that card. When a card asks you to choose (Headbutt, Armaments, ...), the
  same actions mean "pick a card from slot d" instead.
- **END_TURN:** end the turn. The Champ's turn and the start of the next turn happen inside the same step.
- **Potion slot p:** drink one potion of that kind.
- **DONE:** finish an "up to N cards" choice (Purity, Gambling Chip, Elixir).
- Usually only 3–12 actions are legal. `legal_mask` says which.
- "Choose 1 of 3 generated cards" (Discovery, Attack Potion, ...) puts the offers in the `choice` pile.

## Principles

- **Pure functions.** `step(state, action, key)` returns a new state. The same inputs always give the same
  result. All randomness comes from the `key`, so the game never hides a random seed from the search.
- **Fixed shapes, no branches.** Every fight has the same array sizes, so thousands run in parallel.
- **Integers only** in the state and observation.
- **Errors are reported, not hidden.** Problems end the fight with an error flag in the outcome.
- **The sim reports outcomes, the play side decides rewards.**

Details and the reasons behind them: [DESIGN.md](DESIGN.md). Out-of-scope content is listed there too.

## Usage

```
uv sync                          # Python 3.12 + JAX (CPU)
uv sync --extra cuda             # NVIDIA GPU
uv run pytest                    # contract, data checks, oracle scenarios (~45 s; needs an up-to-date oracle)
PARITY_FIGHTS=60 uv run pytest   # random-deck parity fights: a diagnostic for hunting differences
tools/build_oracle.sh            # build the C++ oracle (macOS: may need SDKROOT=.../MacOSX26.5.sdk)
uv run python bench/bench_step.py 4096
uv run python tools/gen_ids.py   # regenerate ids after C++ enum changes
```

```python
import jax, jaxsim

env = jaxsim.make_env()
start = jaxsim.make_start(["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"], hp=80,
                          relics=["BURNING_BLOOD"], potions=["FIRE_POTION"])
state = env.init(start, jax.random.key(0))
state = env.step(state, env.spec.END_TURN, jax.random.key(1))
obs, outcome, legal = env.observe(state), env.outcome(state), env.legal_mask(state)
```

To use it from sts-ml: `uv pip install -e ../sts-lightspeed/jaxsim`.
