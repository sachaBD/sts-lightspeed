"""Branch-free, fixed-shape Champ fight engine (vertical slice).

Layout decisions this slice is meant to test:
  * Cards live in slots (cid/up/cost per slot). Piles are stacks of slot ids
    with a length, so draw/discard are O(1) gathers, not sorts.
  * Effects run through a small micro-op VM over a fixed ring-buffer queue
    (addToTop / addToBot like the C++ ActionQueue), dispatched with lax.switch.
  * The Champ AI is plain arithmetic (no per-monster switch).
  * step(state, action) applies one agent decision, then runs the VM until the
    queue is empty or the fight is over.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import tables as T

N = 64       # card slots (deck + cards created in combat)
H = 10       # hand size
Q = 32       # action queue capacity
C = 96       # relic / power counters and flags
MAX_VM_ITERS = 128

A_END_TURN = H
NUM_ACTIONS = H + 1

i32 = jnp.int32


class State(NamedTuple):
    # cards
    cid: jax.Array        # [N]
    up: jax.Array         # [N]
    cost: jax.Array       # [N] per-slot cost (room for Snecko/Corruption/etc.)
    n_slots: jax.Array
    draw: jax.Array       # [N] slot ids, top = draw[draw_n-1]
    draw_n: jax.Array
    hand: jax.Array       # [H]
    hand_n: jax.Array
    disc: jax.Array       # [N]
    disc_n: jax.Array
    exh: jax.Array        # [N]
    exh_n: jax.Array
    # player
    hp: jax.Array
    max_hp: jax.Array
    block: jax.Array
    energy: jax.Array
    p_status: jax.Array   # [NUM_PLAYER_STATUS]
    # enemy (The Champ)
    e_hp: jax.Array
    e_block: jax.Array
    e_status: jax.Array   # [NUM_ENEMY_STATUS]
    e_move: jax.Array
    e_last: jax.Array
    e_prev: jax.Array
    e_phase2: jax.Array
    e_stances: jax.Array
    # action queue (ring buffer)
    q_op: jax.Array       # [Q]
    q_a: jax.Array
    q_b: jax.Array
    q_front: jax.Array
    q_n: jax.Array
    # misc
    turn: jax.Array
    done: jax.Array
    win: jax.Array
    ctr: jax.Array        # [C] relic/power counters (Pen Nib, Nunchaku, Kunai, ...)
    vm_iters: jax.Array   # instrumentation: VM iterations used by the last step
    key: jax.Array


# ---------------------------------------------------------------- queue helpers
def push_bot(s, op, a=0, b=0):
    i = (s.q_front + s.q_n) % Q
    return s._replace(q_op=s.q_op.at[i].set(op), q_a=s.q_a.at[i].set(a),
                      q_b=s.q_b.at[i].set(b), q_n=s.q_n + 1)


def push_top(s, op, a=0, b=0):
    i = (s.q_front - 1) % Q
    return s._replace(q_op=s.q_op.at[i].set(op), q_a=s.q_a.at[i].set(a),
                      q_b=s.q_b.at[i].set(b), q_front=i, q_n=s.q_n + 1)


def push_bot_if(s, cond, op, a=0, b=0):
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), push_bot(s, op, a, b), s)


def rand_int(s, hi):
    """Uniform int in [0, hi). Returns (state, value)."""
    key, sub = jax.random.split(s.key)
    return s._replace(key=key), jax.random.randint(sub, (), 0, jnp.maximum(hi, 1))


# ---------------------------------------------------------------- piles
def shuffle_discard_into_draw(s):
    key, sub = jax.random.split(s.key)
    idx = jnp.arange(N)
    r = jnp.where(idx < s.disc_n, jax.random.uniform(sub, (N,)), 2.0)
    order = jnp.argsort(r)
    return s._replace(key=key, draw=s.disc[order], draw_n=s.disc_n, disc_n=i32(0))


def draw_one(s):
    s = jax.lax.cond((s.draw_n == 0) & (s.disc_n > 0), shuffle_discard_into_draw, lambda x: x, s)
    ok = (s.draw_n > 0) & (s.hand_n < H)
    slot = s.draw[jnp.maximum(s.draw_n - 1, 0)]
    return s._replace(
        hand=jnp.where(ok, s.hand.at[s.hand_n % H].set(slot), s.hand),
        hand_n=s.hand_n + ok, draw_n=s.draw_n - ok)


def new_slot(s, cid, up):
    slot = jnp.minimum(s.n_slots, N - 1)
    row = cid * 2 + up
    s = s._replace(cid=s.cid.at[slot].set(cid), up=s.up.at[slot].set(up),
                   cost=s.cost.at[slot].set(jnp.asarray(T.CARD_COST)[row]),
                   n_slots=jnp.minimum(s.n_slots + 1, N))
    return s, slot


# ---------------------------------------------------------------- damage
def player_attack_damage(s, base):
    d = base + s.p_status[T.P_STR]
    d = jnp.where(s.p_status[T.P_WEAK] > 0, (d * 3) // 4, d)
    d = jnp.where(s.e_status[T.E_VULN] > 0, (d * 3) // 2, d)
    return jnp.maximum(d, 0)


def enemy_attack_damage(s, base):
    d = base + s.e_status[T.E_STR]
    d = jnp.where(s.e_status[T.E_WEAK] > 0, (d * 3) // 4, d)
    d = jnp.where(s.p_status[T.P_VULN] > 0, (d * 3) // 2, d)
    return jnp.maximum(d, 0)


def hit_player(s, dmg, hits):
    # Per-hit triggers (Flame Barrier, Torii, Tungsten Rod, Buffer...) go here,
    # unrolled over the Champ's max hit count (2).
    total = dmg * hits
    blocked = jnp.minimum(s.block, total)
    hp = s.hp - (total - blocked)
    return s._replace(block=s.block - blocked, hp=hp, done=s.done | (hp <= 0))


# ---------------------------------------------------------------- opcodes
def op_nop(s, a, b):
    return s


def op_attack(s, a, b):
    total = player_attack_damage(s, a) * b
    blocked = jnp.minimum(s.e_block, total)
    e_hp = s.e_hp - (total - blocked)
    won = e_hp <= 0
    return s._replace(e_block=s.e_block - blocked, e_hp=e_hp, done=s.done | won, win=s.win | won)


def op_block(s, a, b):
    g = a + s.p_status[T.P_DEX]
    g = jnp.where(s.p_status[T.P_FRAIL] > 0, (g * 3) // 4, g)
    return s._replace(block=s.block + jnp.maximum(g, 0))


def op_enemy_status(s, a, b):
    return s._replace(e_status=s.e_status.at[a].add(b))


def op_player_status(s, a, b):
    return s._replace(p_status=s.p_status.at[a].add(b))


def op_draw(s, a, b):
    s = draw_one(s)
    return jax.tree.map(lambda x, y: jnp.where(a > 1, x, y), push_top(s, T.OP_DRAW, a - 1), s)


def op_energy(s, a, b):
    return s._replace(energy=s.energy + a)


def op_add_discard(s, a, b):
    s, slot = new_slot(s, a, b)
    return s._replace(disc=s.disc.at[s.disc_n].set(slot), disc_n=s.disc_n + 1)


def op_add_draw_random(s, a, b):
    s, slot = new_slot(s, a, b)
    s, pos = rand_int(s, s.draw_n + 1)
    j = jnp.arange(N)
    shifted = jnp.roll(s.draw, 1)
    draw = jnp.where(j < pos, s.draw, jnp.where(j == pos, slot, shifted))
    return s._replace(draw=draw, draw_n=s.draw_n + 1)


def op_start_turn(s, a, b):
    s = s._replace(turn=s.turn + 1, block=i32(0), energy=i32(3))
    return push_top(s, T.OP_DRAW, 5)


def op_end_turn(s, a, b):
    # Discard the whole hand in order (ethereal/retain hooks would go here).
    j = jnp.arange(N)
    src = j - s.disc_n
    disc = jnp.where((src >= 0) & (src < s.hand_n), s.hand[jnp.clip(src, 0, H - 1)], s.disc)
    s = s._replace(disc=disc, disc_n=s.disc_n + s.hand_n, hand_n=i32(0))
    s = push_bot(s, T.OP_ENEMY_TURN)
    return push_bot(s, T.OP_START_TURN)


def champ_roll(s):
    """The Champ's getMoveForRoll, as arithmetic. Mirrors MonsterSpecific.cpp."""
    s, roll = rand_int(s, 100)
    last, prev = s.e_last, s.e_prev
    below_half = s.e_hp < T.CHAMP_HP // 2
    taunt_turn = ((s.turn + 2) % 4) == 0

    execute = s.e_phase2 & (last != T.CH_EXECUTE) & (prev != T.CH_EXECUTE)
    anger = ~s.e_phase2 & below_half
    taunt = ~s.e_phase2 & ~below_half & taunt_turn
    stance = (roll <= 15) & (last != T.CH_DEFENSIVE_STANCE) & (s.e_stances < 2)
    gloat = (roll <= 30) & (last != T.CH_GLOAT) & (last != T.CH_DEFENSIVE_STANCE)
    slap = (roll <= 55) & (last != T.CH_FACE_SLAP)
    heavy = last != T.CH_HEAVY_SLASH

    move = jnp.select(
        [execute, anger, taunt, stance, gloat, slap, heavy],
        [T.CH_EXECUTE, T.CH_ANGER, T.CH_TAUNT, T.CH_DEFENSIVE_STANCE, T.CH_GLOAT,
         T.CH_FACE_SLAP, T.CH_HEAVY_SLASH],
        T.CH_FACE_SLAP)
    return s._replace(e_move=move, e_prev=last, e_last=move,
                      e_phase2=s.e_phase2 | (move == T.CH_ANGER),
                      e_stances=s.e_stances + (move == T.CH_DEFENSIVE_STANCE))


def op_enemy_turn(s, a, b):
    s = s._replace(e_block=i32(0))
    m = s.e_move
    # Attacks: Heavy Slash 16, Face Slap 12, Execute 10x2.
    base = jnp.select([m == T.CH_HEAVY_SLASH, m == T.CH_FACE_SLAP, m == T.CH_EXECUTE], [16, 12, 10], 0)
    hits = jnp.where(m == T.CH_EXECUTE, 2, jnp.where(base > 0, 1, 0))
    s = hit_player(s, enemy_attack_damage(s, base), hits)
    # Non-attack effects as masked arithmetic.
    es = s.e_status
    es = jnp.where(m == T.CH_ANGER, es.at[T.E_VULN].set(0).at[T.E_WEAK].set(0), es)
    es = es.at[T.E_STR].add(jnp.select([m == T.CH_ANGER, m == T.CH_GLOAT], [6, 3], 0))
    es = es.at[T.E_METALLICIZE].add(jnp.where(m == T.CH_DEFENSIVE_STANCE, 5, 0))
    e_block = jnp.where(m == T.CH_DEFENSIVE_STANCE, 15, 0)
    # End of round: metallicize, debuff decay. Debuffs the Champ applied this
    # round are added after the decay (the game's justApplied flag).
    e_block = e_block + es[T.E_METALLICIZE]
    es = es.at[T.E_VULN].set(jnp.maximum(es[T.E_VULN] - 1, 0)).at[T.E_WEAK].set(jnp.maximum(es[T.E_WEAK] - 1, 0))
    ps = s.p_status
    for k in (T.P_VULN, T.P_WEAK, T.P_FRAIL):
        ps = ps.at[k].set(jnp.maximum(ps[k] - 1, 0))
    ps = ps.at[T.P_FRAIL].add(jnp.where(m == T.CH_FACE_SLAP, 2, 0))
    ps = ps.at[T.P_VULN].add(jnp.where((m == T.CH_FACE_SLAP) | (m == T.CH_TAUNT), 2, 0))
    ps = ps.at[T.P_WEAK].add(jnp.where(m == T.CH_TAUNT, 2, 0))
    s = s._replace(p_status=ps, e_status=es, e_block=e_block)
    s = champ_roll(s)
    return s._replace(done=s.done | (s.turn >= 100))


def op_post_play(s, a, b):
    row = s.cid[a] * 2 + s.up[a]
    t = jnp.asarray(T.CARD_TYPE)[row]
    to_exh = jnp.asarray(T.CARD_EXHAUST)[row] == 1
    to_disc = ~to_exh & (t != T.POWER)
    return s._replace(
        exh=jnp.where(to_exh, s.exh.at[s.exh_n].set(a), s.exh), exh_n=s.exh_n + to_exh,
        disc=jnp.where(to_disc, s.disc.at[s.disc_n].set(a), s.disc), disc_n=s.disc_n + to_disc)


BRANCHES = [op_nop, op_attack, op_block, op_enemy_status, op_player_status, op_draw, op_energy,
            op_add_discard, op_add_draw_random, op_start_turn, op_end_turn, op_enemy_turn, op_post_play]
assert len(BRANCHES) == T.NUM_OPS


def padded_branches(extra, scalar_only=False):
    """Scaling experiment: add `extra` opcodes shaped like real ones (distinct
    constants so XLA cannot merge them) to estimate full-scope VM cost."""
    shapes = [op_attack, op_block, op_draw, op_add_discard, op_end_turn, op_add_draw_random,
              op_player_status, op_post_play]

    if scalar_only:
        shapes = [op_attack, op_block, op_energy]

    def make(k):
        base = shapes[k % len(shapes)]
        if scalar_only:
            return lambda s, a, b: base(s, a + k % 3 + 1, b)
        return lambda s, a, b: base(s._replace(ctr=s.ctr.at[k % C].add(k + 1)), a + k % 3, b)

    return BRANCHES + [make(k) for k in range(extra)]


# ---------------------------------------------------------------- VM
def run_vm(s):
    def cond(c):
        s, it = c
        return (s.q_n > 0) & ~s.done & (it < MAX_VM_ITERS)

    def body(c):
        s, it = c
        i = s.q_front
        op, a, b = s.q_op[i], s.q_a[i], s.q_b[i]
        s = s._replace(q_front=(i + 1) % Q, q_n=s.q_n - 1)
        return jax.lax.switch(op, BRANCHES, s, a, b), it + 1

    s, it = jax.lax.while_loop(cond, body, (s, i32(0)))
    return s._replace(vm_iters=it)


# ---------------------------------------------------------------- public API
def legal_mask(s):
    idx = jnp.arange(H)
    slots = s.hand
    rows = s.cid[slots] * 2 + s.up[slots]
    playable = (idx < s.hand_n) & (jnp.asarray(T.CARD_PLAYABLE)[rows] == 1) & (s.cost[slots] <= s.energy)
    return jnp.concatenate([playable, jnp.array([True])]) & ~s.done


def play_card(s, i):
    slot = s.hand[i]
    row = s.cid[slot] * 2 + s.up[slot]
    j = jnp.arange(H)
    hand = jnp.where(j >= i, jnp.roll(s.hand, -1), s.hand)
    s = s._replace(hand=hand, hand_n=s.hand_n - 1, energy=s.energy - s.cost[slot])
    eff = jnp.asarray(T.CARD_EFFECTS)[row]
    for k in range(T.MAX_EFFECTS):
        s = push_bot_if(s, eff[k, 0] != T.OP_NOP, eff[k, 0], eff[k, 1], eff[k, 2])
    return push_bot(s, T.OP_POST_PLAY, slot)


def step(s, action):
    s = jax.lax.cond(action == A_END_TURN, lambda x: push_bot(x, T.OP_END_TURN),
                     lambda x: play_card(x, action), s)
    return run_vm(s)


def init_state(key, deck_cid, deck_up, player_hp=80):
    """deck_cid/deck_up: int arrays [N], EMPTY-padded."""
    n = jnp.sum(deck_cid != T.EMPTY).astype(i32)
    z = lambda *shape: jnp.zeros(shape, i32)
    key, sub = jax.random.split(key)
    idx = jnp.arange(N)
    order = jnp.argsort(jnp.where(idx < n, jax.random.uniform(sub, (N,)), 2.0)).astype(i32)
    rows = deck_cid * 2 + deck_up
    s = State(
        cid=deck_cid.astype(i32), up=deck_up.astype(i32), cost=jnp.asarray(T.CARD_COST)[rows], n_slots=n,
        draw=order, draw_n=n, hand=z(H), hand_n=i32(0), disc=z(N), disc_n=i32(0), exh=z(N), exh_n=i32(0),
        hp=i32(player_hp), max_hp=i32(player_hp), block=i32(0), energy=i32(0), p_status=z(T.NUM_PLAYER_STATUS),
        e_hp=i32(T.CHAMP_HP), e_block=i32(0), e_status=z(T.NUM_ENEMY_STATUS), e_move=i32(0),
        e_last=i32(T.CH_NONE), e_prev=i32(T.CH_NONE), e_phase2=jnp.array(False), e_stances=i32(0),
        q_op=z(Q), q_a=z(Q), q_b=z(Q), q_front=i32(0), q_n=i32(0),
        turn=i32(0), done=jnp.array(False), win=jnp.array(False), ctr=z(C), vm_iters=i32(0), key=key)
    s = champ_roll(s)
    s = push_bot(s, T.OP_START_TURN)
    return run_vm(s)
