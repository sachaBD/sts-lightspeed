# jaxsim design

**Goal:** the fastest possible simulator of Ironclad vs The Champ, for any deck, built for AlphaZero-style
self-play on one GPU (RTX 3060 or 3090). Search, network and simulator all run inside one jitted program.
The C++ simulator in this repo is the trusted reference.

## Boundaries

- **`jaxsim` (this package) owns the rules:** state, legal actions, transitions, observation and outcome.
  The public contract is `src/jaxsim/api.py`.
- **The play library (sts-ml) owns choices:** search (mctx), network, rewards and targets, replay buffer and
  learner. It sees the simulator only through `jaxsim.Env`.
- The boundary exists only while JAX traces the code, so it costs nothing at runtime: the play library's
  jitted loop traces straight through `Env` functions.

## Core decisions

1. **Pure functions, one game each.** `step(state, action, key) -> state`. Callers use `vmap` and `jit`.
2. **No RNG in the state.** All randomness comes from `key`. Search particles are different keys.
3. **Piles are counts over per-fight card slots.**
   - D=48 slots, one per distinct card identity: card id, upgraded, cost and misc.
   - The observation says which card each slot holds, so slot order carries no meaning and a model
     generalises across decks.
   - Drawing samples from the draw counts. That is equivalent to a uniformly shuffled pile, so the draw
     order is not hidden state.
   - Up to K_TOP=2 known top cards cover put-on-top effects.
   - Created cards reuse a slot with the same identity, or take a free one. If none is free,
     ERR_SLOT_OVERFLOW is set.
4. **Actions line up with identity slots.**
   - Action d plays one copy of card slot d, or picks one during a choice.
   - Then END_TURN, P potion actions and DONE. A = D + P + 2 = 55.
   - Potions are identity slots with counts too, so two Fire Potions are one action. Potion capacity
     (2 / 3 / 5) is a separate field.
   - About 3–12 actions are legal at a time. Masked actions get zero prior in mctx, so the width costs only
     memory.
5. **Choices pick from one pile row with a filter.**
   - The rows are draw, hand, discard, exhaust and PILE_CHOICE. PILE_CHOICE holds generated "choose 1 of 3"
     offers (Discovery, the Attack/Skill/Power/Colorless potions, Nilry's Codex, Toolbox).
   - Picks are sequential, in non-decreasing slot order (the same slot repeats while copies remain).
   - Exactly-N choices end automatically after N picks. DONE ends "up to N" choices.
   - The C++ oracle maps one subset action to the sequence of picks.
6. **Cards are rows of fixed effect columns**, applied by one fused `resolve_card` with masked arithmetic.
   There is no general opcode interpreter and no `while_loop` whose length depends on the data. PR #2 measured
   that every interpreter opcode costs about as much as the arrays it writes, for every lane in the batch.
7. **Relics and powers are masked scalar updates at fixed hook points.** Hot values are separate scalars, not
   entries in shared arrays.
8. **Integers everywhere.** int16 state and observations. Featurisation belongs to the consumer.
9. **No silent failure.** Errors become `Outcome.error` bits and end the game as truncated.

## Out of scope

- Effects that put cards on the bottom of the draw pile (Forethought), and Frozen Eye. Both break the rule
  that draw order is uniformly random given the counts.
- Runic Dome. Cards that generate cards from another character's pool. Other characters' relics and potions.
- Relics that only matter with several enemies or on enemy death (Gremlin Horn, The Specimen, Preserved
  Insect, Sling of Courage), and Smoke Bomb.

`make_start` rejects out-of-scope content. The device side flags ERR_UNSUPPORTED.

## Ids

Ids are generated from the C++ enums by `tools/gen_ids.py`: 0 = NONE, then the C++ names in order. They are
checked in CI against the headers and are part of the contract.

## Layout

```
src/jaxsim/api.py        contract: Env, EnvSpec, StartSpec, Obs, Outcome, CONTRACT, ids, error bits
src/jaxsim/start.py      host-side StartSpec builders
src/jaxsim/_core/        private implementation (state, init, step, observe, cards, champ, piles, choice)
tests/test_contract.py   every promise in api.py
tools/gen_ids.py         id generation and check
bench/bench_step.py      random-policy throughput
```

## Validation

The C++ comparison harness (issue #3) works like this:
1. Export the C++ state.
2. Apply the same action on both sides, injecting the C++'s random outcomes into JAX.
3. Compare the resulting states field by field.

## Roadmap

1. #3 C++ comparison harness.
2. #4 Count-based core with the PR #2 card set and the Champ.
3. sts-ml#10 on-device self-play loop.
4. #5 full card pool.
5. #6 relics and potions.
