"""Exact step-by-step comparison with the C++ oracle on the forced path (C++ draws and Champ rolls fed to JAX).

Used by the scenario tests (gate) and the random-fight parity diagnostic. `Fight` keeps a C++ fight and a JAX
state in lock step; `Fight.act(a)` plays jaxsim action `a` on both and compares the results.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim
from jaxsim._core import init as core_init, step as core_step
from jaxsim._core.rand import Forced
from tests.oracle import convert

SPEC = jaxsim.DEFAULT_SPEC
KEY = jax.random.key(0)
DEBUG = bool(os.environ.get("PARITY_DEBUG"))
_ENV = jaxsim.make_env(SPEC)
_init = jax.jit(lambda start, key, forced: core_init.init(SPEC, start, key, forced))
_step = jax.jit(lambda s, a, key, forced: core_step.step(SPEC, s, a, key, forced))
# one call and one device-to-host copy per step: observation, outcome, legal mask
_view = jax.jit(lambda s: (_ENV.observe(s), _ENV.outcome(s), _ENV.legal_mask(s)))
_obs0 = jax.jit(lambda start: _ENV.observe(core_init.init(SPEC, start, KEY)))  # slot table only


def view(s):
    return jax.device_get(_view(s))


def forced(draw, roll, exhaust=None):
    exhaust = np.zeros(SPEC.D, np.int32) if exhaust is None else exhaust
    return Forced(draw=jnp.asarray(draw, jnp.int32), roll=jnp.int32(roll), exhaust=jnp.asarray(exhaust, jnp.int32))


def roll_of(resp):
    rolls = resp["ai_rolls"]
    assert len(rolls) <= 1, rolls
    return rolls[0] if rolls else 0


def slot_table(obs):
    return " ".join(f"{d}:{jaxsim.CARD_NAMES[int(obs.slot_card[d])]}+{int(obs.slot_upgraded[d])}/c{int(obs.slot_cost[d])}"
                    f"/m{int(obs.slot_misc[d])}[{','.join(str(int(x)) for x in np.asarray(obs.piles)[:4, d])}]"
                    for d in range(SPEC.D) if int(obs.slot_card[d]))


def check(resp, seen, where):
    """Compare the C++ response with the JAX view (obs, outcome, legal mask). Raises AssertionError."""
    obs, out, legal_mask = seen
    if resp["state"]["outcome"] != 0:  # after the fight ends only the result matters (queued leftovers not modelled)
        exp = convert.result(resp["state"])
        act = {"done": bool(out.done), "won": bool(out.won), "player": np.asarray(obs.player)[:2]}
        bad = convert.diff(exp, act)
        assert not bad, f"{where}:\n  " + "\n  ".join(bad)
        return
    try:
        exp = convert.expected(resp["state"], obs, SPEC.D)
    except KeyError as e:
        st = resp["state"]
        cpp = {k: list(st[k]) for k in ("draw", "hand", "discard", "exhaust")}
        raise AssertionError(f"{where}: {e}; jax slots: {slot_table(obs)}; c++ piles {cpp}") from None
    act = convert.actual(obs, out)
    bad = convert.diff(exp, act)
    if bad:
        from jaxsim._core.tables import ids
        nm = lambda cs: [f"{ids.CARDS_CPP[c[0]]}+{c[1]}" for c in cs]
        bad += ["slots: " + slot_table(obs),
                "c++ player: " + str(resp["state"]["player"]) + f" jax known_top {np.asarray(obs.known_top).tolist()}",
                f"c++ drawn: {nm(resp.get('drawn', []))} exhausted: {nm(resp.get('exhausted', []))}"]
    assert not bad, f"{where}:\n  " + "\n  ".join(bad)
    assert int(out.error) == 0, f"{where}: error bits {int(out.error)}"
    if not out.done:
        cpp = set(convert.legal_actions(resp["actions"], obs, SPEC))
        jax_legal = set(np.flatnonzero(np.asarray(legal_mask)).tolist())
        assert cpp == jax_legal, f"{where}: legal c++ {sorted(cpp)} != jax {sorted(jax_legal)}"


class Fight:
    """One fight on both simulators. deck/relics/potions in jaxsim terms (see jaxsim.make_start)."""

    def __init__(self, oracle, name, deck, hp, max_hp=None, relics=(), potions=(), seed=0, enemy_hp=None,
                 setup=None):
        """setup: scenario fields applied after the opening draw on both sides (oracle `set`): enemy_hp,
        enemy_move (move name), energy, block, hp, and piles hand / draw (top first) / discard / exhaust as deck
        entries. Pile cards must be in `deck` (their identities need slots)."""
        self.oracle, self.name = oracle, name
        max_hp = max(hp, 80) if max_hp is None else max_hp
        resp = oracle.start(seed=seed, hp=hp, max_hp=max_hp, deck=convert.cpp_deck(deck),
                            relics=convert.cpp_relics(relics), potions=[convert.CPP_POTION[p] for p in potions])
        start = jaxsim.make_start(deck, hp=hp, max_hp=max_hp, relics=relics, potions=potions)
        obs0 = jax.device_get(_obs0(start))
        hand = convert.slot_counts(obs0, resp["drawn"], SPEC.D)
        self.s = _init(start, KEY, forced(hand, roll_of(resp)))
        setup = dict(setup or {})
        if enemy_hp is not None:  # e.g. a low Champ for killing-blow paths
            setup["enemy_hp"] = enemy_hp
        if setup:
            resp = {**self._set(setup), "drawn": [], "exhausted": []}
        self.resp, self.steps = resp, 0
        self.seen = view(self.s)
        check(resp, self.seen, f"{name} init")

    def _set(self, setup):
        req = {k: v for k, v in setup.items() if k not in ("enemy_move", "hand", "draw", "discard", "exhaust")}
        if "enemy_move" in setup:
            req["enemy_move"] = convert.CPP_MOVE[setup["enemy_move"]]
        piles = [k for k in ("hand", "draw", "discard", "exhaust") if k in setup]
        if piles:
            for k in ("hand", "draw", "discard", "exhaust"):
                req[k] = convert.cpp_deck(setup.get(k, []))
        resp = self.oracle.set(**req)
        i16 = jnp.int16
        fields = {"energy": "energy", "block": "block", "hp": "hp", "enemy_hp": "e_hp"}
        rep = {f: i16(setup[k]) for k, f in fields.items() if k in setup}
        if "enemy_move" in setup:
            rep["e_move"] = i16(convert.MOVE[setup["enemy_move"]])
        if piles:
            obs = view(self.s)[0]
            exp = convert.expected(resp["state"], obs, SPEC.D)["piles"]
            p = np.zeros(np.asarray(obs.piles).shape, np.int16)
            p[:4] = exp
            rep["piles"] = jnp.asarray(p)
            rep["known_top"] = jnp.full_like(self.s.known_top, -1)
        self.s = self.s._replace(**rep)
        return resp

    def find(self, card, up=None) -> int:
        """The legal slot action for a card name (and upgrade), to play or to pick."""
        cid = convert.CARD[card]
        for a in sorted(self.legal()):
            if a < SPEC.D and int(self.obs.slot_card[a]) == cid and (up is None or int(self.obs.slot_upgraded[a]) == up):
                return a
        raise AssertionError(f"{self.name} step {self.steps}: {card}+{up} is not a legal card action; "
                             f"jax slots: {slot_table(self.obs)}")

    def potion(self, name) -> int:
        pid = convert.POTION[name]
        for a in sorted(self.legal()):
            if a > SPEC.END_TURN and int(self.obs.potions[a - SPEC.END_TURN - 1]) == pid:
                return a
        raise AssertionError(f"{self.name} step {self.steps}: potion {name} is not legal")

    @property
    def obs(self):
        return self.seen[0]

    @property
    def done(self):
        return bool(self.seen[1].done)

    def legal(self) -> dict[int, int]:
        """jaxsim action -> C++ bits for the legal actions."""
        try:
            return convert.legal_actions(self.resp["actions"], self.obs, SPEC)
        except KeyError as e:
            raise AssertionError(f"{self.name} step {self.steps}: {e}; jax slots: {slot_table(self.obs)}; "
                                 f"c++ hand {self.resp['state']['hand']}") from None

    def act(self, a: int):
        obs, legal = self.obs, self.legal()
        resp = self.oracle.step(legal[a])
        if DEBUG:
            from jaxsim._core.tables import ids
            print(f"  {self.name} step {self.steps} action {a}: c++ hand",
                  [ids.CARDS_CPP[c[0]] for c in resp["state"]["hand"]], "enemy", resp["state"]["enemy"],
                  "outcome", resp["state"]["outcome"])
        new = {}
        drawn = convert.slot_counts(obs, resp["drawn"], SPEC.D, new=new, ignore_cost=True)
        exhausted = convert.slot_counts(obs, resp["exhausted"], SPEC.D, new=new, ignore_cost=True)
        self.s = _step(self.s, jnp.int32(a), KEY, forced(drawn, roll_of(resp), exhausted))
        self.resp, self.seen = resp, view(self.s)
        check(resp, self.seen, f"{self.name} step {self.steps} action {a}")
        self.steps += 1
