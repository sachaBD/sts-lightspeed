"""The jaxsim contract: everything a consumer (the play library) may depend on.

Consumers import from `jaxsim` (re-exported here) and never from `jaxsim._core`.

Semantics (enforced by tests/test_contract.py):

* **Pure, single-game functions.** Every `Env` function takes and returns JAX pytrees, has no side effects and
  works on ONE game. Callers batch with `jax.vmap` and compile with `jax.jit`; the env never does either.
* **Randomness only from the key argument.** `State` holds no RNG. Identical (state, action, key) give an
  identical result. Search particles are simply different keys: the state carries no hidden information
  besides future randomness, because piles are counts (drawing samples from them) and cards placed on top of
  the draw pile are public (`Obs.known_top`).
* **Opaque state.** `State` is an implementation detail of `_core`; consumers pass it around and read only
  `observe(state)` and `outcome(state)`.
* **One step = one agent decision.** END_TURN runs the enemy turn and the next player turn's start (draw,
  energy, ...) inside the same step.
* **Fixed action layout** (size `spec.A`):
    [0, D)          play the card in slot d, or pick slot d while a choice is pending
    D               END_TURN
    [D+1, D+1+P)    drink one potion of potion slot p
    D+1+P           DONE: finish an "up to N" / "any number" choice
* **Slots are identities with counts.** A card slot is one (card, upgraded, cost, misc) identity; a potion slot
  is one potion id. Copies share a slot and are one action, so the search never branches on identical copies.
  Slot order carries no meaning. A copy whose identity changes (e.g. cost) moves to the matching slot.
* **Choices pick from one pile row.** A pending choice names a source pile (hand, draw, discard, exhaust or
  PILE_CHOICE for generated "choose 1 of 3" offers) and a filter; slot actions pick from it. Picks must be in
  non-decreasing slot order (the same slot again is allowed while copies remain) so every multiset of picks
  has exactly one path. Exactly-N choices end automatically after N picks; DONE ends "up to N" choices.
  END_TURN and potions are illegal while a choice is pending.
* **Terminal states are absorbing.** Once `outcome(s).done`, `step` returns `s` unchanged and `legal_mask`
  allows only END_TURN, so a search always has one legal action.
* **Truncation is not a loss.** The turn cap sets `truncated` (with `won=False`); consumers decide how to score.
* **No silent failure.** Illegal actions, slot overflow and unsupported content set `error` bits and end the
  game with `truncated=True`.
* **Rewards belong to the consumer.** The env reports `Outcome`; it never computes a reward.

Ids (card, relic, potion, move, statuses) are generated from the C++ enums: 0 = NONE, then the C++ names in
order. They are part of the contract. Changing anything in this file, the id tables, or observation semantics
requires bumping `CONTRACT`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax

from jaxsim._core.tables import ids as _ids

CONTRACT = "jaxsim_champ_v1"

__all__ = [
    "CONTRACT", "EnvSpec", "DEFAULT_SPEC", "StartSpec", "Obs", "Outcome", "Env", "make_env",
    "CARD_NAMES", "RELIC_NAMES", "POTION_NAMES", "MOVE_NAMES", "PLAYER_STATUS_NAMES", "MONSTER_STATUS_NAMES",
    "PILE_DRAW", "PILE_HAND", "PILE_DISCARD", "PILE_EXHAUST", "PILE_CHOICE", "NUM_PILES",
    "ERR_ILLEGAL_ACTION", "ERR_SLOT_OVERFLOW", "ERR_INVALID_START", "ERR_UNSUPPORTED", "ERR_QUEUE_OVERFLOW",
]

# Vocabularies: index = jaxsim id. Read-only, exported for consumers (embedding sizes, debugging).
CARD_NAMES: tuple[str, ...] = _ids.CARDS
RELIC_NAMES: tuple[str, ...] = _ids.RELICS
POTION_NAMES: tuple[str, ...] = _ids.POTIONS
MOVE_NAMES: tuple[str, ...] = _ids.MOVES
PLAYER_STATUS_NAMES: tuple[str, ...] = _ids.PLAYER_STATUS
MONSTER_STATUS_NAMES: tuple[str, ...] = _ids.MONSTER_STATUS

# Pile rows of Obs.piles.
PILE_DRAW, PILE_HAND, PILE_DISCARD, PILE_EXHAUST, PILE_CHOICE = range(5)
NUM_PILES = 5  # PILE_CHOICE: cards offered by a pending "choose 1 of N generated cards"

# Error bits of Outcome.error.
ERR_ILLEGAL_ACTION = 1 << 0   # step got an action legal_mask did not allow
ERR_SLOT_OVERFLOW = 1 << 1    # more than D distinct card identities in one fight
ERR_INVALID_START = 1 << 2    # StartSpec failed validation on device
ERR_UNSUPPORTED = 1 << 3      # card / relic / potion not implemented yet
ERR_QUEUE_OVERFLOW = 1 << 4   # internal capacity exceeded


@dataclass(frozen=True)
class EnvSpec:
    """Static sizes. Hashable, so it is safe as a static jit argument. Changing it means recompiling."""

    D: int = 48          # card slots per fight: distinct (card, upgraded, cost, misc) identities incl. generated cards
    H: int = 10          # max hand size
    P: int = 5           # potion slots: distinct potion ids held (capacity is per fight, <= P)
    R: int = 32          # relic slots
    K_TOP: int = 2       # known cards on top of the draw pile
    CHOICE: int = 4      # width of the pending-choice descriptor
    max_turns: int = 50  # truncation cap

    @property
    def END_TURN(self) -> int:
        return self.D

    def potion_action(self, p: int) -> int:
        return self.D + 1 + p

    @property
    def DONE(self) -> int:
        return self.D + 1 + self.P

    @property
    def A(self) -> int:
        return self.D + self.P + 2

    @property
    def n_cards(self) -> int:
        return len(CARD_NAMES)

    @property
    def n_relics(self) -> int:
        return len(RELIC_NAMES)

    @property
    def n_potions(self) -> int:
        return len(POTION_NAMES)

    @property
    def n_moves(self) -> int:
        return len(MOVE_NAMES)

    @property
    def n_player_status(self) -> int:
        return len(PLAYER_STATUS_NAMES)

    @property
    def n_monster_status(self) -> int:
        return len(MONSTER_STATUS_NAMES)


DEFAULT_SPEC = EnvSpec()

# int16 everywhere in StartSpec / Obs / Outcome unless noted. Shapes use EnvSpec names.


class StartSpec(NamedTuple):
    """One fight's starting conditions. Build on the host with `jaxsim.start.make_start`; stack for batches.

    Slots: one row per distinct (card id, upgraded) in the deck, unused rows have card=0 and count=0.
    """

    card: jax.Array        # [D] card id
    upgraded: jax.Array    # [D] 0/1 (Searing Blow: upgrade count)
    count: jax.Array       # [D] copies in the deck
    relics: jax.Array      # [R] relic ids, 0 = empty
    potions: jax.Array     # [P] potion ids, distinct, 0 = empty
    potion_count: jax.Array  # [P] copies held per potion slot
    potion_capacity: jax.Array  # [] potion slots the player has (2 / 3 / 5); sum(potion_count) <= capacity
    hp: jax.Array          # []
    max_hp: jax.Array      # []
    ascension: jax.Array   # []


class Obs(NamedTuple):
    """Public observation. Raw ints; featurisation belongs to the consumer.

    Slot d of every [D] array describes the same card identity, and action d plays/picks that slot, so a
    policy head can produce per-slot logits from per-slot features. Slot order carries no meaning.
    """

    slot_card: jax.Array       # [D] card id (0 = empty slot)
    slot_upgraded: jax.Array   # [D]
    slot_cost: jax.Array       # [D] current cost this turn (-1 = unplayable, -2 = X)
    slot_misc: jax.Array       # [D] per-card value (Rampage / Ritual Dagger / Genetic Algorithm ...)
    piles: jax.Array           # [NUM_PILES, D] counts, rows PILE_*
    known_top: jax.Array       # [K_TOP] slot ids on top of the draw pile, top first; -1 = unknown
    player: jax.Array          # [4] hp, max_hp, block, energy
    player_status: jax.Array   # [n_player_status] amount per status id
    enemy: jax.Array           # [4] hp, max_hp, block, intent move id
    enemy_status: jax.Array    # [n_monster_status] amount per status id
    relics: jax.Array          # [R] relic ids
    relic_counters: jax.Array  # [R] per-relic counter (Pen Nib, Nunchaku, ...), 0 if none
    potions: jax.Array         # [P] potion ids, 0 = empty
    potion_counts: jax.Array   # [P] copies held
    potion_capacity: jax.Array  # []
    choice: jax.Array          # [CHOICE] kind, picks left, source pile, last picked slot; kind 0 = none
    turn: jax.Array            # []


class Outcome(NamedTuple):
    done: jax.Array       # [] bool
    won: jax.Array        # [] bool
    truncated: jax.Array  # [] bool
    hp: jax.Array         # [] int16, player hp now (final hp once done)
    max_hp: jax.Array     # [] int16
    turn: jax.Array       # [] int16
    error: jax.Array      # [] uint32, ERR_* bits


State = Any  # opaque pytree defined in jaxsim._core.state; never inspect it
Action = jax.Array               # [] int32 in [0, spec.A)
Key = jax.Array                  # jax.random.key


@dataclass(frozen=True)
class Env:
    """The simulator as a bundle of pure functions. Pass it to consumers as a static (closed-over) value."""

    spec: EnvSpec
    init: Callable[[StartSpec, Key], State]
    step: Callable[[State, Action, Key], State]
    legal_mask: Callable[[State], jax.Array]   # -> [A] bool
    observe: Callable[[State], Obs]
    outcome: Callable[[State], Outcome]


def make_env(spec: EnvSpec = DEFAULT_SPEC) -> Env:
    from jaxsim._core import env as _env

    return _env.build(spec)
