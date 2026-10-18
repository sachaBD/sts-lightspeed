"""Targeted card rules (parity covers them statistically; these pin known edge cases)."""
import jax
import jax.numpy as jnp
import numpy as np

import jaxsim
from jaxsim import api
from jaxsim._core.state import I16


def _with_hand(env, card, **fields):
    start = jaxsim.make_start([card, "STRIKE_RED"], hp=80)
    s = env.init(start, jax.random.key(0))
    slot = int(np.flatnonzero(np.asarray(s.slot_card) == jaxsim.CARD_NAMES.index(card))[0])
    piles = s.piles.at[api.PILE_HAND].set(0).at[api.PILE_HAND, slot].set(1)
    return s._replace(piles=piles, **{k: I16(v) for k, v in fields.items()}), slot


def test_iron_wave_block_applies_frail_and_dex_once():
    env = jaxsim.make_env()
    for fields, block in ((dict(p_frail=1), 3), (dict(p_dex=2), 7), (dict(p_dex=2, p_frail=1), 5), ({}, 5)):
        s, slot = _with_hand(env, "IRON_WAVE", **fields)
        s = env.step(s, jnp.int32(slot), jax.random.key(1))
        assert int(env.observe(s).player[2]) == block, fields


def _hand_of(env, cards, **fields):
    """State with exactly `cards` (names) in hand and nothing else in piles."""
    start = jaxsim.make_start(cards, hp=80)
    s = env.init(start, jax.random.key(0))
    hand = np.zeros(env.spec.D, np.int16)
    for c in cards:
        hand[int(np.flatnonzero(np.asarray(s.slot_card) == jaxsim.CARD_NAMES.index(c))[0])] += 1
    piles = jnp.zeros_like(s.piles).at[api.PILE_HAND].set(jnp.asarray(hand))
    return s._replace(piles=piles, **{k: I16(v) for k, v in fields.items()})


def _slot(s, card):
    return int(np.flatnonzero(np.asarray(s.slot_card) == jaxsim.CARD_NAMES.index(card))[0])


def _play(env, s, card):
    return env.step(s, jnp.int32(_slot(s, card)), jax.random.key(1))


def test_feed_kill_raises_max_hp_and_wins():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["FEED"], e_hp=5, hp=50), "FEED")
    out = env.outcome(s)
    assert out.won and int(out.max_hp) == 83 and int(out.hp) == 53
    assert int(env.observe(s).piles[api.PILE_EXHAUST, _slot(s, "FEED")]) == 1  # still leaves after the win


def test_reaper_heals_hp_removed_even_on_the_kill():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["REAPER"], e_hp=3, hp=50), "REAPER")
    assert env.outcome(s).won and int(env.outcome(s).hp) == 53
    s = _play(env, _hand_of(env, ["REAPER"], hp=50, energy=3), "REAPER")
    assert int(env.outcome(s).hp) == 54 and int(env.observe(s).enemy[0]) == 440 - 4


def test_clash_needs_only_attacks_in_hand():
    env = jaxsim.make_env()
    assert bool(env.legal_mask(_hand_of(env, ["CLASH", "STRIKE_RED"]))[0:env.spec.D].any())
    s = _hand_of(env, ["CLASH", "DEFEND_RED"])
    assert not bool(env.legal_mask(s)[_slot(s, "CLASH")])


def test_burn_in_hand_hits_through_block_at_end_of_turn():
    env = jaxsim.make_env()
    s = _hand_of(env, ["BURN", "BURN"], block=3)._replace(e_move=I16(0))  # Champ idle: isolate Burn
    s = env.step(s, jnp.int32(env.spec.END_TURN), jax.random.key(1))
    assert int(env.outcome(s).hp) == 80 - (4 - 3)


def test_power_through_wounds_overflow_to_discard():
    env = jaxsim.make_env()
    s = _hand_of(env, ["POWER_THROUGH"] + ["STRIKE_RED"] * 8)
    s = _play(env, s, "POWER_THROUGH")
    w = _slot(s, "WOUND")
    o = env.observe(s)
    assert int(o.piles[api.PILE_HAND].sum()) == 10 and int(o.piles[api.PILE_HAND, w]) == 2
    s = _play(env, _hand_of(env, ["POWER_THROUGH"] + ["STRIKE_RED"] * 9), "POWER_THROUGH")
    w = _slot(s, "WOUND")
    o = env.observe(s)
    assert int(o.piles[api.PILE_HAND, w]) == 1 and int(o.piles[api.PILE_DISCARD, w]) == 1


def test_battle_trance_blocks_further_draws_this_turn():
    env = jaxsim.make_env()
    start = jaxsim.make_start(["BATTLE_TRANCE", "POMMEL_STRIKE"] + ["STRIKE_RED"] * 10, hp=80)
    s = env.init(start, jax.random.key(0))
    bt, ps = _slot(s, "BATTLE_TRANCE"), _slot(s, "POMMEL_STRIKE")
    p = s.piles.at[api.PILE_DRAW].add(s.piles[api.PILE_HAND]).at[api.PILE_HAND].set(0)
    p = p.at[api.PILE_HAND, bt].set(1).at[api.PILE_HAND, ps].set(1).at[api.PILE_DRAW, bt].add(-1).at[api.PILE_DRAW, ps].add(-1)
    s = env.step(s._replace(piles=p), jnp.int32(bt), jax.random.key(1))
    assert int(env.observe(s).piles[api.PILE_HAND].sum()) == 1 + 3
    s = env.step(s, jnp.int32(ps), jax.random.key(2))
    assert int(env.observe(s).piles[api.PILE_HAND].sum()) == 3   # Pommel drew nothing


def test_heavy_blade_applies_strength_three_times():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["HEAVY_BLADE"], p_str=4, energy=3), "HEAVY_BLADE")
    assert int(env.observe(s).enemy[0]) == 440 - (14 + 3 * 4)


def test_perfected_strike_counts_strikes_outside_exhaust_including_itself():
    env = jaxsim.make_env()
    s = _hand_of(env, ["PERFECTED_STRIKE", "STRIKE_RED", "TWIN_STRIKE", "DEFEND_RED"], energy=3)
    p = s.piles.at[api.PILE_DISCARD, _slot(s, "STRIKE_RED")].set(2).at[api.PILE_EXHAUST, _slot(s, "TWIN_STRIKE")].set(5)
    s = _play(env, s._replace(piles=p), "PERFECTED_STRIKE")
    assert int(env.observe(s).enemy[0]) == 440 - (6 + 2 * 5)   # itself + Strike + Twin Strike in hand + 2 discarded


def test_whirlwind_spends_all_energy_and_hits_x_times():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["WHIRLWIND"], energy=3), "WHIRLWIND")
    assert int(env.observe(s).enemy[0]) == 440 - 15 and int(env.observe(s).player[3]) == 0
    s = _hand_of(env, ["WHIRLWIND"], energy=0)
    assert bool(env.legal_mask(s)[_slot(s, "WHIRLWIND")])
    s = _play(env, s, "WHIRLWIND")
    assert int(env.observe(s).enemy[0]) == 440


def test_rampage_grows_per_copy():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["RAMPAGE"], energy=3), "RAMPAGE")
    o = env.observe(s)
    moved = int(np.flatnonzero(np.asarray(o.piles[api.PILE_DISCARD]))[0])
    assert int(o.slot_card[moved]) == jaxsim.CARD_NAMES.index("RAMPAGE") and int(o.slot_misc[moved]) == 5
    assert int(o.enemy[0]) == 440 - 8


def test_blood_for_blood_cheaper_per_hp_losing_hit():
    env = jaxsim.make_env()
    from jaxsim._core import champ
    s = _hand_of(env, ["BLOOD_FOR_BLOOD", "STRIKE_RED"], block=5)._replace(e_move=I16(champ.EXECUTE))
    b = _slot(s, "BLOOD_FOR_BLOOD")
    s = env.step(s, jnp.int32(env.spec.END_TURN), jax.random.key(1))
    assert int(env.observe(s).slot_cost[b]) == 4 - 2    # Execute 10 x 2 through 5 block: both hits lose hp


def test_second_wind_exhausts_sentinel_for_energy():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["SECOND_WIND", "SENTINEL", "DEFEND_RED", "STRIKE_RED"], energy=3), "SECOND_WIND")
    o = env.observe(s)
    assert int(o.player[2]) == 2 * 5 and int(o.player[3]) == 3 - 1 + 2
    assert int(o.piles[api.PILE_HAND].sum()) == 1


def test_sever_soul_kill_leaves_non_attacks_in_hand():
    env = jaxsim.make_env()
    s = _play(env, _hand_of(env, ["SEVER_SOUL", "DEFEND_RED"], e_hp=5, energy=3), "SEVER_SOUL")
    assert env.outcome(s).won and int(env.observe(s).piles[api.PILE_HAND, _slot(s, "DEFEND_RED")]) == 1
