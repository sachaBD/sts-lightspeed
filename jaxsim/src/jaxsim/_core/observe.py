"""observe and outcome: the only views of State that consumers see."""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core.state import I16, State

_PS = {n: i for i, n in enumerate(api.PLAYER_STATUS_NAMES)}
_MS = {n: i for i, n in enumerate(api.MONSTER_STATUS_NAMES)}


def observe(spec: api.EnvSpec, s: State) -> api.Obs:
    ps = jnp.zeros((spec.n_player_status,), I16)
    for name, v in (("STRENGTH", s.p_str), ("DEXTERITY", s.p_dex), ("VULNERABLE", s.p_vuln),
                    ("WEAK", s.p_weak), ("FRAIL", s.p_frail), ("LOSE_STRENGTH", s.p_lose_str),
                    ("NO_DRAW", s.p_no_draw), ("METALLICIZE", s.p_metallicize), ("DEMON_FORM", s.p_demon_form),
                    ("BARRICADE", s.p_barricade), ("BRUTALITY", s.p_brutality), ("COMBUST", s.p_combust),
                    ("CORRUPTION", s.p_corruption), ("DARK_EMBRACE", s.p_dark_embrace), ("EVOLVE", s.p_evolve),
                    ("FEEL_NO_PAIN", s.p_fnp), ("FIRE_BREATHING", s.p_fire_breathing),
                    ("JUGGERNAUT", s.p_juggernaut), ("RAGE", s.p_rage), ("RUPTURE", s.p_rupture),
                    ("FLAME_BARRIER", s.p_flame_barrier), ("INTANGIBLE", s.p_intangible), ("ARTIFACT", s.p_artifact),
                    ("BUFFER", s.p_buffer), ("THORNS", s.p_thorns), ("PLATED_ARMOR", s.p_plated), ("PEN_NIB", s.p_pen_nib),
                    ("NEXT_TURN_BLOCK", s.p_next_turn_block)):
        ps = ps.at[_PS[name]].set(v)
    es = jnp.zeros((spec.n_monster_status,), I16)
    for name, v in (("STRENGTH", s.e_str), ("VULNERABLE", s.e_vuln), ("WEAK", s.e_weak),
                    ("METALLICIZE", s.e_metallicize)):
        es = es.at[_MS[name]].set(v)
    return api.Obs(
        slot_card=s.slot_card, slot_upgraded=s.slot_up, slot_cost=s.slot_cost, slot_misc=s.slot_misc,
        piles=s.piles, known_top=s.known_top,
        player=jnp.stack([s.hp, s.max_hp, s.block, s.energy, s.p_energy_per_turn]), player_status=ps,
        enemy=jnp.stack([s.e_hp, s.e_max_hp, s.e_block, s.e_move, s.e_last, s.e_ai]), enemy_status=es,
        relics=s.relics, relic_counters=s.relic_ctr, potions=s.potions,
        potion_counts=s.potion_n, potion_capacity=s.potion_cap, choice=s.choice, turn=s.turn,
    )


def outcome(spec: api.EnvSpec, s: State) -> api.Outcome:
    return api.Outcome(done=s.done, won=s.won, truncated=s.truncated, hp=s.hp, max_hp=s.max_hp,
                       turn=s.turn, error=s.error)
