# Architecture

## Repos
- **sts-lightspeed** (`sachaBD/sts-lightspeed`): the C++ simulator (the reference) plus `jaxsim/`.
- **sts-ml** (`sachaBD/sts-ml`): the play side.
  - `agents/combat/az/`: model, mctx search, self-play, replay, learner, start banks.
  - `apps/az_train`: config-driven loop, schema `az_champ_v1`.
  - `apps/az_bench`: throughput.
  - jaxsim is installed into sts-ml's `.venv` as an editable package from `$STS_LIGHTSPEED_DIR/jaxsim`.
    The default dir is `../sts_lightspeed`; on the Mac it is `../sts-lightspeed`.
  - `agents/combat/az/model.py` pins `CONTRACT`; a test fails when jaxsim's contract changes.

## The contract (`src/jaxsim/api.py`)
- `Env(spec, init, step, legal_mask, observe, outcome)`: pure, single-game functions. Callers `vmap`/`jit`.
- `CONTRACT = "jaxsim_champ_v1.0"`. Pinned in sts-ml (`AZNet`); no consumer has used it yet, so changes stay v1.0
  until the user says otherwise.
- Actions: `[0, D)` play or pick slot d; `D` END_TURN; `D+1..D+P` potions; `D+P+1` DONE (choices).
- `Obs`:
  - integer arrays: slots (card, upgraded, cost, misc), `piles [5, D]` (draw, hand, discard, exhaust,
    choice), `known_top`;
  - `player [5]` = hp, max hp, block, energy, energy per turn;
  - dense player and monster status arrays indexed by C++ enum names;
  - enemy [6] = hp, max hp, block, intent, previous move, AI state;
  - relics, potions, choice, turn.
- Ids are generated from the C++ headers (`tools/gen_ids.py`): 0 = NONE, then C++ names. `*_CPP` lists map
  raw C++ values for the oracle.

## State model (`_core/state.py`)
- **Card identity slot** = (card, upgraded, cost, misc). Counts per pile. Copies share a slot.
  - A changed copy moves to another identity: Rampage's misc, Blood for Blood's cost, the Corruption cost
    reset.
  - `piles.find_or_alloc(..., want)` takes never-used slots first, then slots with no copies left.
- Drawing = sampling from counts (equivalent to a shuffled pile), so draw order is not hidden state.
- Player powers are separate int16 scalars (`p_*`). The Champ's state is scalars (`e_*`). Maths is done in
  int32, and damage in float32 like the C++.
- `done/won/truncated/error`. Terminal states are absorbing.

## Step pipeline (`_core/step.py`), branch-free
Both the play and END_TURN paths are computed and merged with full-state selects (3 per step: two branch
merges and the illegal/terminal guard). The pipeline follows the C++ action-queue order:
```
play:  cards.play_pre -> draw A (own draws) -> cards.play_post (card leaves, exhaust triggers,
       Juggernaut) -> draw B (Dark Embrace)
end:   end_pre (Metallicize, Burn, Combust, end-of-turn powers, ethereal exhaust, discard)
       -> draw A (Dark Embrace) -> end_mid (champ.take_turn hit by hit, end of round, turn start,
       Brutality) -> draw B (5 + Brutality) -> end_post (Demon Form, energy)
```

## Files
| File | Role |
|---|---|
| `_core/tables/cards.py` | Card table: one row per (card, upgraded) with effect columns + `special` enum + `power` column |
| `_core/cards.py` | `playable`, `play_pre`, `play_post`, damage/block formulas, `exhaust_hand`, powers |
| `_core/hooks.py` | Event helpers: `gain_block` (Juggernaut events), `hurt_enemy`, `lose_hp` (Rupture, Blood for Blood, death), `damage_player_hits`, `on_exhaust` |
| `_core/piles.py` | `find_or_alloc`, vectorised `draw_cards` (one top-k sample; discard copies rank after the draw pile = reshuffle), `draw` with Fire Breathing |
| `_core/champ.py` | A20 Champ: `roll` (AI), `take_turn`, `end_of_round`, `is_attack` |
| `_core/init.py` | StartSpec → State, validation, innate cards, opening draw, first Champ roll |
| `_core/rand.py` | `Forced` (C++ outcomes for parity): draw multiset + Champ roll |
| `_core/observe.py` | State → Obs (scalar statuses scattered into the dense arrays) |
| `start.py` | Host `make_start` (rejects unsupported content), `stack_starts`, `supported_cards()` |
| `oracle/oracle.cpp` | C++ as a JSON-lines process: `start`, `step`, `cards` (card data) |
| `tests/test_parity.py` | Random-deck fights vs the oracle, every step compared |
