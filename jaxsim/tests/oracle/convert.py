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
CPP_POTION = {n: i for i, n in enumerate(ids.POTIONS_CPP)}
POTION = {n: i for i, n in enumerate(jaxsim.POTION_NAMES)}
PILES = ("draw", "hand", "discard", "exhaust")  # api.PILE_* order
ACTION_CARD, ACTION_POTION, ACTION_SELECT, ACTION_END_TURN = 0, 1, 2, 4
OUTCOME_VICTORY, OUTCOME_LOSS = 1, 2


MISC_CARDS = {"RAMPAGE", "RITUAL_DAGGER"}  # cards whose C++ specialData is part of the jaxsim identity (slot_misc)


def identity(c) -> tuple[int, int, int, int]:
    """C++ card json [id, upgrades, cost, costForTurn, misc] -> jaxsim identity (card, upgraded, cost, misc)."""
    name = ids.CARDS_CPP[c[0]]
    return CARD[name], c[1], c[2], c[4] if name in MISC_CARDS else 0


def slot_of(obs, ident, new: dict | None = None, ignore_cost: bool = False) -> int:
    """jaxsim slot of a card identity. An identity created during the step has no slot in `obs` yet; with `new`
    it gets the next never-used slot, as piles.find_or_alloc assigns them. ignore_cost: match a pre-step
    observation whose costs may have changed during the step (Blood for Blood)."""
    card, up, cost, misc = ident
    same_cost = True if ignore_cost else (np.asarray(obs.slot_cost) == cost)
    hit = np.flatnonzero((np.asarray(obs.slot_card) == card) & (np.asarray(obs.slot_upgraded) == up)
                         & same_cost & (np.asarray(obs.slot_misc) == misc))
    if len(hit) > 1 and ignore_cost:   # identities split by cost (Blood for Blood): exact cost, else one with copies
        exact = hit[np.asarray(obs.slot_cost)[hit] == cost]
        piles = np.asarray(obs.piles)
        live = hit[(piles[0, hit] + piles[2, hit]) > 0]
        hit = exact if len(exact) else (live[:1] if len(live) else hit[:1])
    if len(hit) == 1:
        return int(hit[0])
    if len(hit) == 0 and new is not None:
        if ident not in new:
            empty = np.flatnonzero(np.asarray(obs.slot_card) == 0)
            new[ident] = int(empty[len(new)])
        return new[ident]
    raise KeyError(f"no unique slot for {jaxsim.CARD_NAMES[card]}+{up} cost {cost} misc {misc}")


def slot_counts(obs, cards, D, new: dict | None = None, ignore_cost: bool = False) -> np.ndarray:
    out = np.zeros(D, np.int32)
    for ident, n in Counter(identity(c) for c in cards).items():
        out[slot_of(obs, ident, new, ignore_cost)] += n
    return out


def cpp_deck(deck):
    """jaxsim deck list (names or (name, up)) -> oracle deck [[cpp id, upgrades]]."""
    return [[CPP_CARD[c], 0] if isinstance(c, str) else [CPP_CARD[c[0]], *c[1:]] for c in deck]


def cpp_relics(relics):
    """jaxsim relics [(name, data)] -> oracle relics [[cpp id, data]]."""
    return [[CPP_RELIC[n], d] for n, d in relics]


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
        "player": np.array([p["hp"], p["max_hp"], p["block"], p["energy"], p["energy_per_turn"]]),
        "player_status": ps,
        "enemy": np.array([e["hp"], e["max_hp"], e["block"], MOVE[ids.MOVES_CPP[e["move"]]],
                           MOVE[ids.MOVES_CPP[e["last"]]], e["misc"]]),
        "enemy_status": es,
        "turn": np.array(state["turn"]),
        "potion_counts": potion_counts(state, obs),
        "done": state["outcome"] != 0,
        "won": state["outcome"] == OUTCOME_VICTORY,
    }


def potion_counts(state, obs) -> np.ndarray:
    out = np.zeros(len(np.asarray(obs.potions)), np.int32)
    for cpp in state["potions"]:
        name = ids.POTIONS_CPP[cpp]
        if name in ("INVALID", "EMPTY_POTION_SLOT"):
            continue
        out[int(np.flatnonzero(np.asarray(obs.potions) == POTION[name])[0])] += 1
    return out


def actual(obs, out) -> dict:
    return {
        "potion_counts": np.asarray(obs.potion_counts),
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
        if a["type"] in (ACTION_CARD, ACTION_SELECT):   # play a hand card / pick a card for a choice
            out.setdefault(slot_of(obs, identity(a["card"])), a["bits"])
        elif a["type"] == ACTION_END_TURN:
            out[spec.END_TURN] = a["bits"]
        elif a["type"] == ACTION_POTION:
            if a.get("discard"):
                continue                                   # jaxsim has no potion discard
            pid = POTION[ids.POTIONS_CPP[a["potion"]]]
            slot = int(np.flatnonzero(np.asarray(obs.potions) == pid)[0])
            out.setdefault(spec.potion_action(slot), a["bits"])
        else:
            raise NotImplementedError(f"oracle action type {a['type']}")
    return out
