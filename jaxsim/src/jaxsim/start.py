"""Host-side helpers that build StartSpec arrays. Raises on bad input (the device side can only flag errors)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxsim import api

_CARD = {n: i for i, n in enumerate(api.CARD_NAMES)}
_RELIC = {n: i for i, n in enumerate(api.RELIC_NAMES)}
_POTION = {n: i for i, n in enumerate(api.POTION_NAMES)}


def _id(table: dict[str, int], name: str, kind: str) -> int:
    if name not in table or table[name] == 0:
        raise ValueError(f"unknown {kind}: {name!r}")
    return table[name]


def make_start(
    deck: Iterable[str | tuple[str, int]],
    *,
    hp: int,
    max_hp: int | None = None,
    relics: Sequence[str] = (),
    potions: Sequence[str] = (),
    ascension: int = 0,
    spec: api.EnvSpec = api.DEFAULT_SPEC,
) -> api.StartSpec:
    """deck: card names ("STRIKE_RED") or (name, upgraded) pairs. Identical cards share one slot."""
    counts = Counter((c, 0) if isinstance(c, str) else (c[0], int(c[1])) for c in deck)
    if len(counts) > spec.D:
        raise ValueError(f"deck has {len(counts)} distinct cards, more than D={spec.D}")
    if len(relics) > spec.R or len(potions) > spec.P:
        raise ValueError("too many relics or potions for spec")
    max_hp = hp if max_hp is None else max_hp
    if not 0 < hp <= max_hp:
        raise ValueError(f"bad hp {hp}/{max_hp}")

    card = np.zeros(spec.D, np.int16)
    up = np.zeros(spec.D, np.int16)
    count = np.zeros(spec.D, np.int16)
    for i, ((name, u), n) in enumerate(sorted(counts.items())):
        card[i], up[i], count[i] = _id(_CARD, name, "card"), u, n
    rel = np.zeros(spec.R, np.int16)
    rel[: len(relics)] = [_id(_RELIC, r, "relic") for r in relics]
    pot = np.zeros(spec.P, np.int16)
    pot[: len(potions)] = [_id(_POTION, p, "potion") for p in potions]
    i16 = lambda x: jnp.asarray(x, jnp.int16)
    return api.StartSpec(card=i16(card), upgraded=i16(up), count=i16(count), relics=i16(rel),
                         potions=i16(pot), hp=i16(hp), max_hp=i16(max_hp), ascension=i16(ascension))


def stack_starts(starts: Sequence[api.StartSpec]) -> api.StartSpec:
    """Stack single starts into a batched StartSpec (leading axis = batch) for vmap'd init."""
    return jax.tree.map(lambda *x: jnp.stack(x), *starts)
