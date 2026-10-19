"""step and legal_mask.

One step = one decision. Both decision kinds are computed and selected (branch-free under vmap), around two
shared draws that follow the C++ action-queue order:

    play card:  play_pre  -> draw A (the card's own draws) -> play_post (card leaves, exhaust triggers,
                Juggernaut) -> draw B (Dark Embrace / Evolve draws)
    END_TURN:   end_pre (Metallicize, Burn, Combust, end-of-turn powers, ethereal exhaust, discard)
                -> draw A (Dark Embrace draws from ethereal exhaust) -> end_mid (Champ turn, end of round,
                next turn start, Brutality) -> draw B (5 + Brutality + Evolve) -> Demon Form, energy

Ethereal-exhaust Dark Embrace draws trigger Evolve after the Champ's turn here (the C++ queues those Evolve
draws behind the monster turn start); only Fire Breathing timing on such a draw could differ.
"""
import jax
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import cards, champ, choice, hooks, piles
from jaxsim._core.state import I16, I32, U32, State, tree_where
from jaxsim._core.tables import cards as tc
from jaxsim._core.tables import relics as relics_tr

HAND_DRAW = 5
i16 = hooks.i16


def legal_mask(spec: api.EnvSpec, s: State):
    """Normal: playable cards and END_TURN. Choice pending: the valid picks only. Terminal: END_TURN only."""
    pending = (s.choice[choice.KIND] > 0) & ~s.done
    slots = jnp.where(pending, choice.pick_mask(s), cards.playable(s)) & ~s.done
    mask = jnp.zeros((spec.A,), bool).at[: spec.D].set(slots)
    return mask.at[spec.END_TURN].set(~pending)


def end_pre(spec: api.EnvSpec, s: State, key, forced):
    """Up to the discard. Returns (state, Dark Embrace draws, late Juggernaut events, forced)."""
    r = cards.rows(s)
    # Orichalcum (no block), Metallicize, then Plated Armor block (Juggernaut)
    from jaxsim._core import relics
    s, j_or = hooks.gain_block(s, jnp.where(relics.has(s, "ORICHALCUM") & (s.block <= 0), 6, 0), ~s.done)
    s, jugg = hooks.gain_block(s, s.p_metallicize.astype(I32), ~s.done)
    jugg = jugg + j_or
    s, j_pl = hooks.gain_block(s, s.p_plated.astype(I32), ~s.done)
    jugg = jugg + j_pl
    # end-of-turn cards in hand (Burn): blockable self damage, one hit per Burn
    hand = s.piles[api.PILE_HAND].astype(I32)
    burns = jnp.where(r[:, tc.col("eot_damage")] > 0, hand, 0)
    per_burn = jnp.max(jnp.where(burns > 0, r[:, tc.col("eot_damage")], 0))   # Burn+ only from other enemies
    s = hooks.damage_player_hits(s, per_burn, burns.sum(), True, ~s.done)
    # Doubt (Weak), Shame (Frail): monster-sourced debuffs; Regret: lose hp = cards in hand, per Regret
    go = ~s.done
    s = champ._debuff_player(s, "p_weak", champ.JA_WEAK, jnp.where(go, jnp.sum(hand * r[:, tc.col("eot_weak")]), 0))
    s = champ._debuff_player(s, "p_frail", champ.JA_FRAIL, jnp.where(go, jnp.sum(hand * r[:, tc.col("eot_frail")]), 0))
    regrets = jnp.sum(hand * r[:, tc.col("eot_regret")])
    per = jnp.where(s.p_intangible > 0, jnp.minimum(hand.sum(), 1), hand.sum())
    s = hooks.lose_hp(s, regrets * per, True, ~s.done & (regrets > 0), events=regrets)
    # Combust: lose 1 hp per Combust played (self damage), then damage the enemy
    s = hooks.lose_hp(s, s.p_combust_hp.astype(I32), True, ~s.done & (s.p_combust > 0))
    s = hooks.hurt_enemy(s, s.p_combust.astype(I32), ~s.done & (s.p_combust > 0))
    # end-of-turn powers: Flex's strength loss, No Draw and Rage expire
    # (the strength loss is DebuffPlayer<STRENGTH>(-n): Artifact negates it)
    from jaxsim._core import relics
    lose = jnp.where(~s.done, s.p_lose_str.astype(I32), 0)
    s = relics.debuff_player(s, "p_str", 0, -lose)
    s = s._replace(p_lose_str=I16(0), p_no_draw=I16(0), p_rage=I16(0))
    # draws triggered by hp loss so far this end of turn (Runic Cube, Centennial Puzzle) happen now, before the
    # discard (the C++ adds them to the top of the queue)
    s, forced, _, _ = piles.draw(spec, s, jnp.where(s.done, 0, s.p_pending_draw.astype(I32)), key, forced, extend=False)
    s = s._replace(p_pending_draw=I16(0))
    # ethereal cards exhaust (with triggers), the rest of the hand is discarded
    s, _, j2, de = cards.exhaust_hand(s, r[:, tc.col("ethereal")] == 1, ~s.done)
    p = s.piles.astype(I32)
    p = p.at[api.PILE_DISCARD].add(p[api.PILE_HAND]).at[api.PILE_HAND].set(0)
    s = s._replace(piles=p.astype(I16))
    s = hooks.juggernaut(s, jugg)
    # Juggernaut from the ethereal-exhaust Feel No Pain block lands after the Champ's block reset (queued behind
    # the monster turn start); carried into end_mid
    return s, de, j2, forced


def end_mid(spec: api.EnvSpec, s: State, roll, jugg_late, fb_late):
    """Champ turn, end of round, next turn start. Returns (state, cards to draw)."""
    played_last_turn = s.p_cards_played
    s = champ.take_turn(s, roll, jugg_late, fb_late)
    go = ~s.done
    nxt = champ.end_of_round(s)
    w = lambda new, old: jnp.where(go, new, old)
    cap = go & (s.turn + 1 >= spec.max_turns)
    s = s._replace(
        e_block=w(nxt.e_block, s.e_block), e_vuln=w(nxt.e_vuln, s.e_vuln), e_weak=w(nxt.e_weak, s.e_weak),
        e_ja=w(nxt.e_ja, s.e_ja),
        p_vuln=w(nxt.p_vuln, s.p_vuln), p_weak=w(nxt.p_weak, s.p_weak), p_frail=w(nxt.p_frail, s.p_frail),
        p_ja=w(nxt.p_ja, s.p_ja), turn=w(i16(s.turn + 1), s.turn),
        block=w(jnp.where(s.p_barricade > 0, s.block,
                          jnp.where(s.rel[relics_tr.ID["CALIPERS"]], jnp.maximum(s.block - 15, 0), 0)).astype(I16), s.block),
        p_flame_barrier=w(I16(0), s.p_flame_barrier), p_cards_played=w(I16(0), s.p_cards_played),
        p_intangible=w(nxt.p_intangible, s.p_intangible), done=s.done | cap, truncated=s.truncated | cap)
    from jaxsim._core import relics
    s, energy_bonus = relics.turn_start(s)
    # Self-Forming Clay's block; Pocketwatch (3 cards or fewer last turn: draw 3 more)
    s, ev = hooks.gain_block(s, s.p_next_turn_block.astype(I32), go)
    s = hooks.juggernaut(s, ev)._replace(p_next_turn_block=jnp.where(go, I16(0), s.p_next_turn_block))
    watch = go & relics.has(s, "POCKETWATCH") & (played_last_turn <= 3)
    s = s._replace(p_attacks_played=jnp.where(go, I16(0), s.p_attacks_played),
                   p_skills_played=jnp.where(go, I16(0), s.p_skills_played))
    # Brutality: lose hp (not self damage) and draw, after the hand draw
    s = hooks.lose_hp(s, s.p_brutality.astype(I32), False, ~s.done & (s.p_brutality > 0))
    n = HAND_DRAW + s.p_brutality.astype(I32) + 3 * watch + s.p_pending_draw.astype(I32)
    s = s._replace(p_pending_draw=I16(0))
    return s, jnp.where(s.done, 0, n), energy_bonus


def end_post(s: State, is_end, energy_bonus) -> State:
    go = is_end & ~s.done
    return s._replace(p_str=i16(s.p_str + jnp.where(go, s.p_demon_form, 0)),
                      energy=jnp.where(go, i16(jnp.where(s.rel[relics_tr.ID["ICE_CREAM"]], s.energy, 0)
                                               + s.p_energy_per_turn + energy_bonus), s.energy))


def step(spec: api.EnvSpec, s: State, action, key, forced=None) -> State:
    action = jnp.asarray(action, I32)
    a = jnp.clip(action, 0, spec.A - 1)
    ok = (action >= 0) & (action < spec.A) & legal_mask(spec, s)[a]
    is_end = a == spec.END_TURN
    d = jnp.clip(a, 0, spec.D - 1)
    k_a, k_b, k_roll = jax.random.split(key, 3)
    roll = jax.random.randint(k_roll, (), 0, 100) if forced is None else forced.roll

    k_ex, k_a = jax.random.split(k_a)
    pending0 = s.choice[choice.KIND] > 0
    limbo0 = s.choice[choice.LIMBO].astype(I32)
    kind = cards.row(s, d)[tc.col("choice")]

    # play a card: effects, then a pre-draw choice (everything but Warcry) that may stay pending
    played, n_own, trig, jugg, relic_draw = cards.play_pre(spec, s, d, k_ex, forced)
    # (a pending choice stores relic draws queued behind the card, e.g. Ink Bottle: they come after the pick)
    played, pend_a, jugg, trig = choice.choice_point(spec, played, jnp.where(kind != tc.CH_WARCRY, kind, 0), d,
                                                     jugg, trig + relic_draw)
    trig = trig - relic_draw
    n_own = n_own + relic_draw
    def queued_after_choice(st, slot, go):
        """Rage and the on-use relic actions of a card with a choice: after the choice resolves."""
        rr = cards.row(s, slot)
        go = go & (rr[tc.col("choice")] != tc.CH_NONE)
        st, ev1 = rage(st, slot, go)
        st, ev2, draw = cards.relic_effects(st, rr[tc.col("type")], go)
        return st, ev1 + ev2, draw

    rage = lambda st, slot, go: hooks.gain_block(st, jnp.where(
        go & (cards.row(s, slot)[tc.col("type")] == tc.ATTACK) & (cards.row(s, slot)[tc.col("choice")] != tc.CH_NONE),
        st.p_rage.astype(I32), 0), ~st.done)
    played, ev, choice_draw = queued_after_choice(played, d, ~pend_a & (kind != tc.CH_WARCRY))
    jugg = jugg + ev
    n_own = n_own + choice_draw
    # pick for a pending choice: its effect, then the rest of the card (Burning Pact draws after the pick)
    picked, j_p, t_p = choice.apply_pick(spec, s, s.choice[choice.KIND].astype(I32), d, limbo0, ~s.done)
    picked = picked._replace(choice=jnp.zeros_like(s.choice))
    picked, ev_p, pick_draw = queued_after_choice(picked, limbo0, True)
    j_p = j_p + ev_p
    lim_kind = s.choice[choice.KIND].astype(I32)
    n_pick = jnp.where(lim_kind != tc.CH_WARCRY, cards.row(s, limbo0)[tc.col("draw")], 0) + pick_draw
    acted = tree_where(pending0, picked, played)
    jugg = jnp.where(pending0, s.choice[choice.JUGG] + j_p, jugg)
    trig = jnp.where(pending0, s.choice[choice.TRIG] + t_p, trig)
    n_a = jnp.where(pending0, n_pick, jnp.where(pend_a, 0, n_own))
    limbo = jnp.where(pending0, limbo0, d)

    k_end, k_a = jax.random.split(k_a)
    ended, n_de, jugg_late, forced_end = end_pre(spec, s, k_end, forced)
    mid = tree_where(is_end, ended, acted)
    forced = forced if forced is None else tree_where(is_end, forced_end, forced)
    mid, forced, evolve_later, fb_late = piles.draw(spec, mid, jnp.where(is_end, n_de, n_a), k_a, forced,
                                                    extend=False, fire_now=~is_end)
    # Warcry's choice comes after its own draws
    post_kind = jnp.where(~pending0 & ~pend_a & ~is_end & (kind == tc.CH_WARCRY), kind, 0)
    # (a pending Warcry choice stores the Evolve draws its own draws triggered; they happen after the pick)
    mid, pend_b, jugg, trig = choice.choice_point(spec, mid, post_kind, d, jugg, trig + evolve_later)
    trig = trig - evolve_later
    pending = ~pending0 & (pend_a | pend_b) & ~is_end
    # Warcry resolved now: its queued relic actions (Ink Bottle's draw joins the later draw)
    mid, ev_w, warcry_draw = queued_after_choice(mid, d, ~pending0 & ~is_end & (kind == tc.CH_WARCRY) & ~pend_b)
    jugg = jugg + ev_w
    trig = trig + warcry_draw
    evolve_later = jnp.where(pend_b, 0, evolve_later)

    left, trig2 = cards.play_post(spec, mid, limbo, jugg, ~pending)
    turned, n_hand, energy_bonus = end_mid(spec, mid, roll, jugg_late, fb_late)
    mid = tree_where(is_end, turned, left)
    n_b = jnp.where(is_end, n_hand, jnp.where(pending, 0, trig + trig2 + mid.p_pending_draw)) + evolve_later
    mid = mid._replace(p_pending_draw=jnp.where(is_end | pending, mid.p_pending_draw, I16(0)))
    k_b, k_top = jax.random.split(k_b)
    mid, forced, _, _ = piles.draw(spec, mid, jnp.where(mid.done, 0, n_b), k_b, forced, extend=True)
    # Unceasing Top: an empty hand during the turn draws a card
    top = (~is_end & ~pending & ~mid.done & mid.rel[relics_tr.ID["UNCEASING_TOP"]]
           & (mid.piles[api.PILE_HAND].sum() == 0))
    mid, _, _, _ = piles.draw(spec, mid, jnp.where(top, 1, 0), k_top, forced, extend=True)
    nxt = end_post(mid, is_end, energy_bonus)

    # the C++ can't-win rule (executeActions): no cards in hand, draw or discard and no damage without cards
    # (Combust, Juggernaut) is a loss
    p = nxt.piles.astype(I32)
    no_cards = (p[api.PILE_HAND].sum() + p[api.PILE_DRAW].sum() + p[api.PILE_DISCARD].sum()) == 0
    stuck = no_cards & ~nxt.done & (nxt.p_combust == 0) & (nxt.p_juggernaut == 0) & (nxt.choice[choice.KIND] == 0)
    nxt = nxt._replace(done=nxt.done | stuck)

    # one full select: illegal actions and terminal states keep s; then patch flags
    take = ok & ~s.done
    out = tree_where(take, nxt, s)
    bad = ~ok & ~s.done
    return out._replace(done=out.done | bad, truncated=out.truncated | bad,
                        error=out.error | jnp.where(bad, U32(api.ERR_ILLEGAL_ACTION), U32(0)))
