"""Map C++ oracle states into jaxsim terms (by name) and compare them with jaxsim observations."""
from collections import Counter

import numpy as np

import jaxsim
from jaxsim._core.tables import ids

CARD = {n: i for i, n in enumerate(jaxsim.CARD_NAMES)}
MOVE = {n: i for i, n in enumerate(jaxsim.MOVE_NAMES)} | {"INVALID": 0}  # C++ sentinel -> NONE
PS = {n: i for i, n in enumerate(jaxsim.PLAYER_STATUS_NAMES)}
MS = {n: i for i, n in enumerate(jaxsim.MONSTER_STATUS_NAMES)}
CPP_CARD = {n: i for i, n in enumerate(ids.CARDS_CPP)}
CPP_RELIC = {n: i for i, n in enumerate(ids.RELICS_CPP)}
PILES = ("draw", "hand", "discard", "exhaust")  # api.PILE_* order
ACTION_CARD, ACTION_END_TURN = 0, 4
OUTCOME_VICTORY, OUTCOME_LOSS = 1, 2


def identity(c) -> tuple[int, int]:
    """C++ card json [id, upgrades, cost, costForTurn, misc] -> jaxsim (card id, upgraded)."""
    return CARD[ids.CARDS_CPP[c[0]]], c[1]


def slot_of(obs, ident) -> int:
    card, up = ident
    hit = np.flatnonzero((np.asarray(obs.slot_card) == card) & (np.asarray(obs.slot_upgraded) == up))
    if len(hit) != 1:
        raise KeyError(f"no unique slot for {jaxsim.CARD_NAMES[card]}+{up}")
    return int(hit[0])


def slot_counts(obs, cards, D) -> np.ndarray:
    out = np.zeros(D, np.int32)
    for ident, n in Counter(identity(c) for c in cards).items():
        out[slot_of(obs, ident)] += n
    return out


def cpp_deck(deck):
    """jaxsim deck list (names or (name, up)) -> oracle deck [[cpp id, upgrades]]."""
    return [[CPP_CARD[c], 0] if isinstance(c, str) else [CPP_CARD[c[0]], c[1]] for c in deck]


def expected(state, obs, D) -> dict:
    """The C++ state as the jaxsim observation fields it should equal."""
    p, e = state["player"], state["enemy"]
    ps = np.zeros(len(jaxsim.PLAYER_STATUS_NAMES), np.int32)
    for cpp, v in p["status"]:
        ps[PS[ids.PLAYER_STATUS_CPP[cpp]]] = v
    es = np.zeros(len(jaxsim.MONSTER_STATUS_NAMES), np.int32)
    for cpp, v in e["status"]:
        es[MS[ids.MONSTER_STATUS_CPP[cpp]]] = v
    return {
        "piles": np.stack([slot_counts(obs, state[name], D) for name in PILES]),
        "player": np.array([p["hp"], p["max_hp"], p["block"], p["energy"]]),
        "player_status": ps,
        "enemy": np.array([e["hp"], e["max_hp"], e["block"], MOVE[ids.MOVES_CPP[e["move"]]],
                           MOVE[ids.MOVES_CPP[e["last"]]], e["misc"]]),
        "enemy_status": es,
        "turn": np.array(state["turn"]),
        "done": state["outcome"] != 0,
        "won": state["outcome"] == OUTCOME_VICTORY,
    }


def actual(obs, out) -> dict:
    return {
        "piles": np.asarray(obs.piles)[:4], "player": np.asarray(obs.player),
        "player_status": np.asarray(obs.player_status), "enemy": np.asarray(obs.enemy),
        "enemy_status": np.asarray(obs.enemy_status), "turn": np.asarray(obs.turn),
        "done": bool(out.done), "won": bool(out.won),
    }


_NAMES = {"player_status": jaxsim.PLAYER_STATUS_NAMES, "enemy_status": jaxsim.MONSTER_STATUS_NAMES}


def _show(k, v):
    v = np.asarray(v)
    if k in _NAMES:
        return {_NAMES[k][i]: int(v[i]) for i in np.flatnonzero(v)}
    return v.tolist()


def diff(exp: dict, act: dict) -> list[str]:
    return [f"{k}: c++ {_show(k, exp[k])} != jax {_show(k, act[k])}"
            for k in exp if not np.array_equal(np.asarray(exp[k]), np.asarray(act[k]))]


def legal_actions(actions, obs, spec) -> dict[int, int]:
    """C++ legal actions -> {jaxsim action: C++ bits}. Identical cards collapse onto one slot action."""
    out = {}
    for a in actions:
        if a["type"] == ACTION_CARD:
            out.setdefault(slot_of(obs, identity(a["card"])), a["bits"])
        elif a["type"] == ACTION_END_TURN:
            out[spec.END_TURN] = a["bits"]
        else:
            raise NotImplementedError(f"oracle action type {a['type']}")
    return out
