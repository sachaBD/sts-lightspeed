# jaxsim: Champ fight vertical slice

This is a prototype of a table-driven, fixed-shape JAX engine for Ironclad vs
The Champ. Its job is to measure whether the design is fast enough before the
full card, relic and potion list gets built. Scope and design are in
[DESIGN.md](DESIGN.md).

What the slice includes:
- 11 cards: Strike, Defend, Bash, Pommel Strike, Shrug It Off, Inflame, Twin
  Strike, Iron Wave, Anger, Wild Strike, Wound.
- The full Champ AI: all 7 moves, phase 2, Defensive Stance limit, Taunt
  timing.
- Turn structure, draw, shuffle, discard and exhaust piles, a micro-op
  action queue, and a legal-action mask.

What it doesn't include yet:
- Relics, potions, card selection, the card replay queue.
- Bit-exact RNG.
- Validation against the C++ simulator.

## Run

```
pip install "jax[cuda12]"      # or "jax[cpu]"
python bench.py 4096 16384 65536
EXTRA_OPS=30 python bench.py 4096                    # extra opcodes that write arrays
EXTRA_OPS=60 PAD_KIND=scalar python bench.py 4096    # extra opcodes that touch scalars only
```

## Results

Measured on a 4-core cloud CPU, not a GPU. The policy picks uniformly random
legal actions, and fights reset automatically when they end.

| Configuration | Batch | Decisions/s | Fights/s |
|---|---|---|---|
| Slice, 13 opcodes | 4096 | 33.5k | 1.1k |
| + 8 array-writing opcodes, approximating full scope (21) | 4096 | 19.8k | 0.66k |
| same | 16384 | 28.4k | 0.94k |
| + 30 array-writing opcodes (43) | 4096 | 9.4k | 0.31k |
| + 60 array-writing opcodes (73) | 4096 | 5.7k | 0.19k |
| + 60 **scalar-only** opcodes (73) | 4096 | 33.3k | 1.1k |

Each decision used 4.1 VM iterations on average. The batch maximum, which
`vmap` pays for, was 8, so the while-loop wastes about 2×.

**Design rule from these results.** Under `vmap`, each opcode costs about as
much as the arrays it writes. Opcodes that only touch scalars are close to
free.
- Keep array-writing opcodes (pile moves, card creation, card selection) to
  about 10–15.
- Express everything else, including relics, powers and statuses, as masked
  scalar arithmetic.
- Store hot counters as separate scalars, not entries in a shared array.
