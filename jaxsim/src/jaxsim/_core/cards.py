"""Card play: legality and the fused resolve over the card table's fixed effect columns.

A play is split around the step's two draw points (step.py), matching the C++ queue order:
    play_pre   energy, self hp loss, exhaust-first effects (Fiend Fire, Second Wind), block, damage, specials,
               debuffs, buffs, powers, energy, Rage block          -> the card's own draws
    play_post  created cards, Battle Trance, the card leaving (exhaust triggers), Juggernaut damage
                                                                     -> triggered draws (Dark Embrace, Evolve)
Returns carry the draw counts; Juggernaut events accumulate in the state-free return value.
"""
import jax.numpy as jnp

from jaxsim import api
from jaxsim._core import champ, hooks, piles
from jaxsim._core.state import I16, I32, State, tree_where
from jaxsim._core.tables import cards as tc

TABLE = jnp.asarray(tc.TABLE)
F32 = jnp.float32
C = tc.col
i16 = hooks.i16
from jaxsim._core.tables import relics as _tr
_VELVET = _tr.ID["VELVET_CHOKER"]

# power column -> state field (BARRICADE / CORRUPTION set a flag; BERSERK is handled separately)
_ADD = {"metallicize": "p_metallicize", "demon_form": "p_demon_form", "brutality": "p_brutality",
        "combust": "p_combust", "dark_embrace": "p_dark_embrace", "evolve": "p_evolve", "fnp": "p_fnp",
        "fire_breathing": "p_fire_breathing", "juggernaut": "p_juggernaut", "rage": "p_rage",
        "rupture": "p_rupture", "flame_barrier": "p_flame_barrier"}
_FLAG = {"barricade": "p_barricade", "corruption": "p_corruption"}


def rows(s: State):
    """[D, n_columns] table rows of every slot."""
    return TABLE[s.slot_card.astype(I32), jnp.minimum(s.slot_up, 1).astype(I32)]


def row(s: State, d):
    return TABLE[s.slot_card[d].astype(I32), jnp.minimum(s.slot_up[d], 1).astype(I32)]


def effective_cost(s: State, r, cost):
    """Corruption makes skills free."""
    return jnp.where((s.p_corruption > 0) & (r[..., C("type")] == tc.SKILL), 0, cost)


def playable(s: State):
    r = rows(s)
    cost = effective_cost(s, r, s.slot_cost.astype(I32))
    hand = s.piles[api.PILE_HAND] > 0
    only_attacks = jnp.sum(jnp.where(hand & (r[:, C("type")] != tc.ATTACK), 1, 0)) == 0
    clash_ok = (r[:, C("special")] != tc.SP_CLASH) | only_attacks
    affordable = (cost == -1) | ((cost >= 0) & (cost <= s.energy))   # -1 = X: always playable
    normality = jnp.any(hand & (r[:, C("normality")] == 1)) & (s.p_cards_played >= 3)
    normality = normality | (s.rel[_VELVET] & (s.p_cards_played >= 6))
    return hand & (r[:, C("playable")] == 1) & affordable & clash_ok & ~normality


def _has(s: State, name: str):
    return s.rel[_tr.ID[name]]


def player_attack_damage(s: State, base, strength, strike=False):
    """BattleContext::calculateCardDamage: float32 chain, truncated once. Strike Dummy +3 on strike cards, Pen Nib
    x2, Weak x0.75, Vulnerable x1.5 (x1.75 with Paper Phrog)."""
    d = base.astype(F32) + jnp.where(_has(s, "STRIKE_DUMMY") & strike, F32(3), F32(0)) + strength.astype(F32)
    d = jnp.where(s.p_pen_nib > 0, d * F32(2), d)
    d = jnp.where(s.p_weak > 0, d * F32(0.75), d)
    d = jnp.where(s.e_vuln > 0, d * jnp.where(_has(s, "PAPER_PHROG"), F32(1.75), F32(1.5)), d)
    return jnp.maximum(d.astype(I32), 0)


def attack_enemy(s: State, per, hits, go) -> State:
    """`hits` attacks of `per` damage (Monster::attacked): block absorbs hit by hit; an unblocked part of 1-4 becomes
    5 with The Boot."""
    per, hits = jnp.asarray(per, I32), jnp.where(go, jnp.asarray(hits, I32), 0)
    b = s.e_block.astype(I32)
    full = jnp.where(per > 0, jnp.minimum(hits, b // jnp.maximum(per, 1)), hits)   # hits fully blocked
    rest_block = b - full * per
    left = hits - full
    first = jnp.where(left > 0, per - jnp.minimum(rest_block, per), 0)             # the first hit through
    boot = lambda u: jnp.where(_has(s, "THE_BOOT") & (u > 0) & (u < 5), 5, u)
    unblocked = boot(first) + jnp.maximum(left - 1, 0) * boot(per)
    s = s._replace(e_block=i16(s.e_block - jnp.minimum(b, per * hits)))   # 0 left if any hit got through
    return hooks.hurt_enemy(s, unblocked, go & (hits > 0) & (unblocked > 0))


def card_block(s: State, base):
    """BattleContext::calculateCardBlock: integer maths."""
    b = jnp.where(s.p_dex != 0, jnp.maximum(base + s.p_dex.astype(I32), 0), base)
    return jnp.where(s.p_frail > 0, b * 3 // 4, b)


def exhaust_hand(s: State, mask, go):
    """Exhaust every hand copy of the slots in `mask`, with on-exhaust triggers.
    Returns (state, number exhausted, Juggernaut events, triggered draws)."""
    return exhaust_counts(s, jnp.where(mask, s.piles[api.PILE_HAND].astype(I32), 0), go)


def returns_on_exhaust(s: State, n, go) -> State:
    """Necronomicurse / Curse of the Bell: an exhausted copy puts a new copy in hand (discard if the hand is full)."""
    ret = jnp.where(go, n * rows(s)[:, C("returns_on_exhaust")], 0)
    p = s.piles.astype(I32)
    room = jnp.maximum(10 - p[api.PILE_HAND].sum(), 0)
    to_hand = _take_first_c(ret, room)
    p = p.at[api.PILE_HAND].add(to_hand).at[api.PILE_DISCARD].add(ret - to_hand)
    s = s._replace(piles=p.astype(I16))
    # the returned card is a new, unupgraded copy (MakeTempCardInHand(id))
    move = jnp.zeros_like(s.piles, I32).at[api.PILE_HAND].set(to_hand).at[api.PILE_DISCARD].set(ret - to_hand)
    card = s.slot_card.astype(I32)
    return piles.retarget(s, move, card, jnp.zeros_like(card), TABLE[card, 0, C("cost")], jnp.zeros_like(card))


def _take_first_c(c, k):
    return jnp.clip(k - (jnp.cumsum(c) - c), 0, c)


def exhaust_counts(s: State, n, go):
    """Exhaust n[d] hand copies of each slot, with on-exhaust triggers."""
    p = s.piles.astype(I32)
    n = jnp.where(go, n, 0)
    p = p.at[api.PILE_HAND].add(-n).at[api.PILE_EXHAUST].add(n)
    s = returns_on_exhaust(s._replace(piles=p.astype(I16)), n, go)
    p = s.piles.astype(I32)
    s = s._replace(piles=p.astype(I16))
    s, jugg, draws = hooks.on_exhaust(s, n.sum(), jnp.sum(n * rows(s)[:, C("exhaust_energy")]), go)
    return s, n.sum(), jugg, draws


def play_pre(spec: api.EnvSpec, s: State, d, key, forced):
    """Returns (state, own draws, triggered draws, Juggernaut events, relic draws queued behind the card)."""
    r = row(s, d)
    sp, ctype = r[C("special")], r[C("type")]
    hand = s.piles[api.PILE_HAND]
    x = s.energy.astype(I32)                                            # X for X-cost cards
    cost = effective_cost(s, r, s.slot_cost[d].astype(I32))
    spent = jnp.where(cost == -1, x, cost)
    s = s._replace(energy=i16(s.energy - spent), piles=s.piles.at[api.PILE_HAND, d].set(hand[d] - 1),
                   p_cards_played=i16(s.p_cards_played + 1),
                   p_attacks_played=i16(s.p_attacks_played + (ctype == tc.ATTACK)))
    str_at_play = s.p_str                                               # damage is precomputed at play
    # Pain in hand: 1 hp per Pain, before the card resolves (addToTop in triggerOnOtherCardPlayed)
    pains = jnp.sum(jnp.where(rows(s)[:, C("pain")] == 1, s.piles[api.PILE_HAND].astype(I32), 0))
    s = hooks.lose_hp(s, pains, False, ~s.done & (pains > 0), events=pains)

    # self hp loss (no block, Rupture)
    s = hooks.lose_hp(s, r[C("hp_loss")], True, ~s.done)
    go = ~s.done

    # exhaust before the effect: Fiend Fire (whole hand), Second Wind (non-attacks)
    rs = rows(s)
    non_attack = rs[:, C("type")] != tc.ATTACK
    ex_mask = jnp.where(sp == tc.SP_FIEND_FIRE, True, jnp.where(sp == tc.SP_SECOND_WIND, non_attack, False))
    s, n_exhausted, jugg, trig = exhaust_hand(s, ex_mask, go)

    # block: one GainBlock per card (Second Wind: one per exhausted card); Entrench doubles current block
    per = jnp.where(r[C("block")] > 0, card_block(s, r[C("block")]), 0)
    body_slam_block = s.block
    blocks = jnp.where(sp == tc.SP_SECOND_WIND, n_exhausted, (per > 0).astype(I32))
    s, ev = hooks.gain_block(s, per * blocks, go)
    jugg = jugg + jnp.where(sp == tc.SP_SECOND_WIND, blocks * (per > 0) * go, ev)
    s, ev = hooks.gain_block(s, jnp.where(sp == tc.SP_ENTRENCH, s.block, 0), go)
    jugg = jugg + ev
    # Armaments+: upgrade every upgradable card in hand. True Grit: exhaust a random card in hand.
    from jaxsim._core import choice
    s = choice.upgrade_copies(s, jnp.where((sp == tc.SP_UPGRADE_HAND) & go & choice.can_upgrade(s),
                                           s.piles[api.PILE_HAND].astype(I32), 0))
    hand_now = s.piles[api.PILE_HAND].astype(I32)
    if forced is None:
        pick, _ = piles._sample(hand_now, jnp.zeros_like(hand_now), 1, key)
    else:
        pick = jnp.minimum(forced.exhaust.astype(I32), hand_now)
    s, _, j2, t2 = exhaust_counts(s, jnp.minimum(pick, 1), go & (sp == tc.SP_RANDOM_EXHAUST))
    jugg, trig = jugg + j2, trig + t2

    # damage x hits (same per-hit damage; block absorbs hit by hit, which sums to this)
    dropkick_bonus = (sp == tc.SP_DROPKICK) & (s.e_vuln > 0)          # checked before the hit
    p = s.piles.astype(I32)
    strikes = jnp.sum(rs[:, C("is_strike")] * (p[api.PILE_HAND] + p[api.PILE_DRAW] + p[api.PILE_DISCARD]))
    n_up = s.slot_up[d].astype(I32)
    amt = r[C("special_amt")]
    base = jnp.select(
        [sp == tc.SP_HEAVY_BLADE, sp == tc.SP_PERFECTED_STRIKE, sp == tc.SP_BODY_SLAM, sp == tc.SP_SEARING_BLOW,
         sp == tc.SP_RAMPAGE, sp == tc.SP_MISC_DAMAGE, sp == tc.SP_MIND_BLAST],
        [r[C("damage")] + amt * str_at_play, r[C("damage")] + amt * (strikes + 1), body_slam_block,
         n_up * (n_up + 7) // 2 + 12, r[C("damage")] + s.slot_misc[d], s.slot_misc[d], p[api.PILE_DRAW].sum()],
        r[C("damage")])
    hits = jnp.select([sp == tc.SP_WHIRLWIND, sp == tc.SP_FIEND_FIRE], [x, n_exhausted], r[C("hits")])
    per_hit = player_attack_damage(s, base, str_at_play, r[C("is_strike")] == 1)
    e_hp_before = s.e_hp
    s = attack_enemy(s, per_hit, hits, go & (hits > 0))
    # Reaper heals the hp it removed; Feed raises max hp on the kill (both survive the win)
    healed = jnp.where((sp == tc.SP_REAPER) & go, e_hp_before - s.e_hp, 0) + jnp.where(go, r[C("heal")], 0)
    feed = jnp.where((sp == tc.SP_FEED) & s.won & go, r[C("special_amt")], 0)
    s = s._replace(max_hp=i16(s.max_hp + feed))
    s = hooks.heal(s, healed + feed, healed + feed > 0)

    # everything below is cleared on victory
    live = ~s.done
    s, _, j2, t2 = exhaust_hand(s, (sp == tc.SP_SEVER_SOUL) & non_attack, live)   # queued behind its attack
    jugg, trig = jugg + j2, trig + t2
    p_str = s.p_str + r[C("p_str")] * live
    p_str = p_str + jnp.where((sp == tc.SP_SPOT_WEAKNESS) & champ.is_attack(s.e_move) & live, r[C("special_amt")], 0)
    p_str = jnp.where((sp == tc.SP_LIMIT_BREAK) & live, p_str * 2, p_str)
    belt = (r[C("e_vuln")] > 0) & live & _has(s, "CHAMPION_BELT")   # Champion Belt: Weak 1 with each Vulnerable
    s = s._replace(e_vuln=i16(s.e_vuln + r[C("e_vuln")] * live), e_weak=i16(s.e_weak + r[C("e_weak")] * live + belt),
                   e_str=i16(s.e_str + r[C("e_str")] * live), p_str=i16(p_str),

                   energy=i16(s.energy + (r[C("energy")] + dropkick_bonus) * live),
                   p_intangible=i16(s.p_intangible + jnp.where((sp == tc.SP_INTANGIBLE) & live, r[C("special_amt")], 0)))
    from jaxsim._core import relics
    s = relics.debuff_player(s, "p_lose_str", 0, r[C("lose_str")] * live)    # Flex (Artifact negates it)
    s = _apply_power(s, r, live)
    # Rage: block after each attack's actions (after its choice, if it has one: see step.py)
    rage = (ctype == tc.ATTACK) & (r[C("choice")] == tc.CH_NONE)
    s, ev = hooks.gain_block(s, jnp.where(rage, s.p_rage.astype(I32), 0), ~s.done)
    jugg = jugg + ev
    # relic counters now; their effects now unless the card has a choice (then after it: step.py)
    s = relic_counters(s, ctype)
    s, ev, ink_draw = relic_effects(s, ctype, r[C("choice")] == tc.CH_NONE)
    s = hooks.heal(s, 2, (ctype == tc.POWER) & _has(s, "BIRD_FACED_URN") & ~s.done)   # immediate p.heal
    jugg = jugg + ev
    n_draw = (r[C("draw")] + dropkick_bonus) * live
    return s, n_draw, trig, jugg, ink_draw


def relic_counters(s: State, ctype) -> State:
    """onUse*Card relic counters, updated when the card is played."""
    go = ~s.done
    attack, skill, power = (ctype == tc.ATTACK) & go, (ctype == tc.SKILL) & go, (ctype == tc.POWER) & go
    pen = s.r_pen_nib + (attack & _has(s, "PEN_NIB"))
    nun = s.r_nunchaku + (attack & _has(s, "NUNCHAKU"))
    ink = s.r_ink + ((attack | skill | power) & _has(s, "INK_BOTTLE"))
    return s._replace(r_pen_nib=i16(jnp.where(pen == 9, -1, pen)), r_nunchaku=i16(jnp.where(nun >= 10, 0, nun)),
                      r_ink=i16(jnp.where(ink == 10, 0, ink)), p_skills_played=i16(s.p_skills_played + skill))


def relic_effects(s: State, ctype, go):
    """The actions those relics queue behind the card (after its choice, if it has one); triggers are read back
    from the counters. Returns (state, Juggernaut events, draws)."""
    go = go & ~s.done
    attack, skill, power = (ctype == tc.ATTACK) & go, (ctype == tc.SKILL) & go, (ctype == tc.POWER) & go
    third = attack & (s.p_attacks_played % 3 == 0)
    s = s._replace(p_dex=i16(s.p_dex + (third & _has(s, "KUNAI"))), p_str=i16(s.p_str + (third & _has(s, "SHURIKEN"))))
    s, ev = hooks.gain_block(s, jnp.where(third & _has(s, "ORNAMENTAL_FAN"), 4, 0), go)
    # Pen Nib: this attack used the status; the 10th attack gets it (counter -1 right after it triggers)
    s = s._replace(p_pen_nib=i16(jnp.where(attack, _has(s, "PEN_NIB") & (s.r_pen_nib == -1), s.p_pen_nib)))
    s = s._replace(energy=i16(s.energy + (attack & _has(s, "NUNCHAKU") & (s.r_nunchaku == 0))))
    skills = s.p_skills_played
    s = hooks.hurt_enemy(s, 5, skill & _has(s, "LETTER_OPENER") & (skills >= 3) & (skills % 3 == 0))
    ink = (attack | skill | power) & _has(s, "INK_BOTTLE") & (s.r_ink == 0)
    return s, ev, ink.astype(I32)


def _apply_power(s: State, r, live) -> State:
    pw, amt = r[C("power")], r[C("power_amt")] * live
    upd = {}
    for name, field in _ADD.items():
        upd[field] = i16(getattr(s, field) + jnp.where(pw == tc.PW[name], amt, 0))
    for name, field in _FLAG.items():
        upd[field] = i16(jnp.where((pw == tc.PW[name]) & live, 1, getattr(s, field)))
    combust = (pw == tc.PW["combust"]) & live
    berserk = (pw == tc.PW["berserk"]) & live
    upd["p_combust_hp"] = i16(s.p_combust_hp + combust)
    upd["p_energy_per_turn"] = i16(s.p_energy_per_turn + berserk)

    # CardManager::onBuffCorruption: the first Corruption sets every skill's cost to 0 for the combat
    first_corruption = (pw == tc.PW["corruption"]) & live & (s.p_corruption == 0)
    skill = rows(s)[:, C("type")] == tc.SKILL
    upd["slot_cost"] = i16(jnp.where(first_corruption & skill & (s.slot_cost > 0), 0, s.slot_cost))
    s = s._replace(**upd)
    from jaxsim._core import champ, relics
    return relics.debuff_player(s, "p_vuln", champ.JA_VULN, jnp.where(berserk, amt, 0), monster_source=False)


def play_post(spec: api.EnvSpec, s: State, d, jugg, active=True):
    """Created cards and Battle Trance (cleared on victory), the card leaving (unless the player died), then
    Juggernaut damage. Nothing happens unless `active` (a choice is pending). Returns (state, triggered draws)."""
    r = row(s, d)
    live = ~s.done & active
    p = s.piles.astype(I32)
    p = p.at[api.PILE_DISCARD, d].add(r[C("copy_to_discard")] * live)
    s = s._replace(piles=p.astype(I16), p_no_draw=(s.p_no_draw + ((r[C("special")] == tc.SP_BATTLE_TRANCE) & live))
                   .astype(I16))
    # created cards: draw pile (shuffled in), discard pile, or hand (overflowing to discard at 10)
    card, pile, n = r[C("create_card")], r[C("create_pile")], r[C("create_n")]
    want = (card > 0) & live
    s, slot, ok = piles.find_or_alloc(s, card, 0, TABLE[card, 0, C("cost")], 0, want)
    n = jnp.where(want & ok, n, 0)
    p = s.piles.astype(I32)
    room = jnp.maximum(spec.H - p[api.PILE_HAND].sum(), 0)
    to_hand = jnp.where(pile == tc.TO_HAND, jnp.minimum(n, room), 0)
    to_discard = jnp.where(pile == tc.TO_DISCARD, n, 0) + jnp.where(pile == tc.TO_HAND, n - to_hand, 0)
    to_draw = jnp.where(pile == tc.TO_DRAW, n, 0)
    p = (p.at[api.PILE_HAND, slot].add(to_hand).at[api.PILE_DISCARD, slot].add(to_discard)
          .at[api.PILE_DRAW, slot].add(to_draw))
    s = s._replace(piles=p.astype(I16))
    # the played card leaves: powers vanish, exhausting cards (and skills under Corruption) go to exhaust with
    # their triggers, the rest to discard. After a win it still leaves; after the player dies it stays in limbo.
    exhausts = (r[C("exhausts")] == 1) | ((s.p_corruption > 0) & (r[C("type")] == tc.SKILL))
    leaves = (r[C("type")] != tc.POWER) & (s.hp > 0) & active
    rampage = (r[C("special")] == tc.SP_RAMPAGE) & active   # leaves with its bonus raised: a new identity
    s, slot2, ok2 = piles.find_or_alloc(s, s.slot_card[d], s.slot_up[d], s.slot_cost[d],
                                        s.slot_misc[d] + r[C("special_amt")], rampage)
    leave = jnp.where(rampage & ok2, slot2, d)
    p = s.piles.astype(I32)
    p = p.at[jnp.where(exhausts, api.PILE_EXHAUST, api.PILE_DISCARD), leave].add(leaves.astype(I32))
    s = s._replace(piles=p.astype(I16))
    s, j2, trig = hooks.on_exhaust(s, (exhausts & leaves).astype(I32),
                                   jnp.where(exhausts & leaves, r[C("exhaust_energy")], 0), ~s.done)
    s = hooks.juggernaut(s, jnp.where(active, jugg + j2, 0))
    return s, trig * (~s.done) * active
