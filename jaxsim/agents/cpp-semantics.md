# C++ semantics that are easy to get wrong

Read the C++ before implementing anything: `src/combat/BattleContext.cpp` (card `case CardId::X`),
`src/combat/Actions.cpp` (action bodies), `src/combat/Player.cpp` (powers and hooks), `CardManager.cpp`
(draw, exhaust bookkeeping), `MonsterSpecific.cpp` (Champ), `include/constants/Cards.h` (costs, exhaust,
ethereal, innate, strike tags). Then let parity tell you what you got wrong.

## Action queue
- Card actions are queued addToBot in order. `OnAfterCardUsed` (the card leaving play) is queued **after**
  them, during `useCard`.
- Triggers queued while actions run land **behind** `OnAfterCardUsed`: Juggernaut damage, Dark Embrace and
  Evolve draws. That is why a step has two draw points.
- `Action.clearOnCombatVictory`:
  - false for GainBlock, PlayerLoseHp, HealPlayer, DamagePlayer, OnAfterCardUsed, so they still run after the
    killing blow;
  - true for everything else (damage, debuffs, draws, created cards).
- If the player dies from their own card (Offering), the card stays in limbo (it never reaches a pile).
- `AttackEnemy(t, dmg)` damage is computed **at play time** (Hemokinesis doesn't see its own Rupture
  Strength). `AttackAllEnemy` computes at resolution.
- Sever Soul queues its exhausts behind its attack, so a killing Sever Soul exhausts nothing. Second Wind and
  Fiend Fire exhaust first (addToTop).

## Numbers
- Card damage: a float32 chain (base + Strength, ×0.75 Weak, ×1.5 Vulnerable), truncated once.
- Block: integer maths with `*3/4` for Frail.
- Monster damage to the player is computed once per move; then each hit goes through block.
- A20 Champ:
  - 440 HP; Heavy Slash 18; Face Slap 14 + 2 Frail + 2 Vulnerable; Execute 10×2;
  - Gloat +5 Strength; Anger: clear debuffs, then +12 Strength;
  - Defensive Stance: 20 block + 7 Metallicize, on a roll ≤ 30, at most twice (`miscInfo & 3`);
  - Taunt when `(turn + 2) % 4 == 0`, giving 2 Weak + 2 Vulnerable;
  - phase 2 starts when HP < max/2 at roll time (bit 4 of `miscInfo`). In phase 2 he Executes unless it was
    used in either of the last two moves.
- Monster damage to an attacking player uses `moveHistory[0]` = intent, `[1]` = previous.

## Status quirks
- A debuff from a monster is "just applied" (skips that round's decrement) **only if the player didn't already
  have it**.
- Player debuffs (Vulnerable, Weak, Frail) decrement at end of round; monster Vulnerable/Weak too (no
  just-applied from player cards).
- No Draw stacks. Berserk has no status: it raises `energyPerTurn`.
- Barricade is a flag with no `statusMap` entry (the oracle reports 1).
- The first Corruption permanently sets every skill's cost to 0 in all piles (`onBuffCorruption`). Later
  skills cost 0 while it's active, and skills exhaust when played.
- Blood for Blood gets 1 cheaper per hp-loss event (`hpWasLost`): each hit through block, each Burn, self
  damage.
- Rupture: Strength on self-damage hp loss (card hp loss, Burn, Combust; not Brutality).
- Perfected Strike counts strike-tagged cards in hand + draw + discard + itself (not exhaust).
- Innate cards (Brutality+) start on top of the draw pile; more than 5 innate means extra draws.
- Ethereal exhaust at end of turn triggers Feel No Pain and Dark Embrace. Those land after the discard and
  **before** the monster turn, and the drawn cards stay in hand.
- Flame Barrier hits back after each monster hit (even fully blocked) and can kill the Champ on his turn.

## Choices and queue timing (batch 4)
- A choice pauses the action queue (`InputState::CARD_SELECT`). Everything queued behind it runs after the pick:
  Burning Pact's draws, Rage block (onUseAttackCard), OnAfterCardUsed, and queued Juggernaut/Dark Embrace.
- Auto-resolve rules differ per action (ArmamentsAction, HeadbuttAction, ChooseExhaustOne, WarcryAction,
  DualWieldAction, ExhumeAction): count valid **copies**; 0 does nothing, 1 resolves, 2+ opens the select.
- End-of-turn triggers from ethereal exhaust are queued behind UnnamedEndOfTurnAction, which queues
  MonsterStartTurn. Their follow-ups (Juggernaut damage, Fire Breathing on drawn statuses) land after the
  monster's block reset.
- The can't-win rule in `executeActions`: no cards anywhere → loss unless something deals damage without cards.
- `UpgradeAllCardsInHand` (Armaments+) upgraded curses too (fixed). Upgrading resets cost to the upgraded
  table cost when it differs (Blood for Blood keeps no reduction; Corruption-zeroed skills regain a cost).

## Oracle gotchas
- `PlayerStatus` starts with INVALID = 0 and ends with THE_BOMB. `MonsterStatus` has INVALID last. Loop
  bounds matter.
- Relic bits only cover ids < 128 (higher ids alias).
- C++ costs: −1 = X, −2 = unplayable, −3 Ascender's Bane.
- Potion "discard" actions exist in C++ legal actions; jaxsim has no discard.

## C++ bugs fixed in both simulators
Iron Wave block calculated twice; Disarm+ was −2; Seeing Red didn't exhaust; Rage cost 1 (fell through
`getEnergyCost`'s default); Armaments+ upgraded curses; the can't-win check ignored Combust and Juggernaut;
8 card colours rotated (Brilliance/Brutality/Buffer/Bullet Time,
Collect/Combust/Compile Driver/Concentrate). `tests/test_cpp_tables.py` and the card-table parity test guard
against this kind of drift.
