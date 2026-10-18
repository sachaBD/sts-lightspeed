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

## Game settings

- Ascension 20 only.
- Bottled cards are out of scope.

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
     (2 at A20, 4 with Potion Belt) is a separate field.
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

`tests/test_parity.py` plays 200 random-deck fights against the C++ simulator (the `jaxsim_oracle` process,
built by `tools/build_oracle.sh`) and compares every step: piles, player, enemy, statuses, turn, outcome and
the legal-action set. Once the fight is over only the result is compared (done, won, hp, max hp): effects still
queued after the end (e.g. a Dark Embrace draw after the killing blow) are not modelled.
- The C++'s random outcomes are injected into JAX through the private forced path (`_core/rand.py`):
  - drawn cards: the hand difference before and after;
  - Champ rolls: replayed from a copy of `aiRng`.
- Half the fights use a uniform random policy. The other half always play cards and start at 300 HP, so they
  reach phase 2, Execute and wins.
- Coverage: `uv run python -m tests.oracle.coverage`.

C++ behaviour that jaxsim copies on purpose (the C++ is the reference):
- A monster's debuff on the player is "just applied" (skips that round's decrement) only if the player didn't
  already have it.
- When a card wins the fight, its later effects are skipped (Bash's Vulnerable, Anger's copy, draws), but the
  card itself still goes to the discard pile.
- When the player dies during the Champ's turn, everything stops there: no debuffs, no roll, no end of round.
- After a winning hit, block, HP loss, healing and the played card leaving still happen; other effects don't
  (C++ `clearOnCombatVictory`). If the player dies from their own card (Offering), the card stays in limbo.
- No Draw stacks (two Battle Trances give No Draw 2). Ascender's Bane has cost −3.
- Sever Soul attacks before exhausting (its exhausts are queued behind the attack), so a killing Sever Soul
  exhausts nothing. Fiend Fire and Second Wind exhaust first.
- Blood for Blood gets 1 cheaper per hp-loss event: each Champ hit that gets through block, each Burn, each
  self-damage card. Perfected Strike counts Strike cards in hand, draw and discard plus itself, not exhaust.
- Step order follows the C++ action queue (see `_core/step.py`):
  - playing a card: its effects, its own draws, the card leaving (exhaust triggers), Juggernaut, then triggered
    draws (Dark Embrace, Evolve);
  - END_TURN: Metallicize, Burn, Combust, end-of-turn powers, ethereal exhaust (Feel No Pain, Dark Embrace),
    discard, Dark Embrace draws, the Champ's turn hit by hit (Flame Barrier hits back after each), end of
    round, next turn start (Barricade, draw 5, Brutality, Demon Form, energy).
  - Evolve draws triggered by an end-of-turn Dark Embrace draw happen after the Champ's turn (the C++ queues
    them behind the monster turn start); only Fire Breathing timing on such a draw could differ.
- The first Corruption sets every skill's cost to 0 for the combat (CardManager::onBuffCorruption); later
  skills are free while it is active. Berserk raises energy per turn (no status in the C++): Obs.player[4].
- Card identity includes cost (Blood for Blood) and misc (Rampage's bonus). A played Rampage leaves to a new
  identity with its bonus raised. Slots whose copies are all gone are reused once no never-used slot is left.

## Supported content

- **Cards** (`jaxsim.supported_cards()`), upgraded and unupgraded:
  - starter cards, PR #2's set, batch 1 of #5 (table rows and small columns) and batch 2 (formulas:
    Heavy Blade, Perfected Strike, Body Slam, Whirlwind, Searing Blow, Rampage, Blood for Blood, Sever Soul,
    Second Wind, Fiend Fire) and batch 3 (powers and hooks: Metallicize, Demon Form, Barricade, Berserk,
    Brutality, Combust, Corruption, Dark Embrace, Evolve, Feel No Pain, Fire Breathing, Juggernaut, Rupture,
    Rage, Flame Barrier);
  - the statuses those cards create: Wound, Dazed, Burn;
  - Ascender's Bane.
- No relic effects and no potions yet. Relics are carried in the observation only.
- `make_start` rejects anything else.
- The card table's cost, type, exhausts and ethereal values are checked against the C++ card data (`test_card_table_matches_cpp`).

## Measurements

Random policy, `bench/bench_step.py`, MacBook Air CPU:

| Batch | Decisions/s | Fights/s |
|---|---|---|
| 1024 | 429k | 18.3k |
| 4096 | 444k | 19.0k |
| 16384 | 428k | 18.4k |

## Roadmap

1. #3 C++ comparison harness.
2. #4 Count-based core with the PR #2 card set and the Champ.
3. sts-ml#10 on-device self-play loop.
4. #5 full card pool.
5. #6 relics and potions.
