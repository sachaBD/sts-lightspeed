# JAX Champ fight: scope and table-driven design

Goal: RL with a value network on **Ironclad vs The Champ**, arbitrary deck,
all Ironclad-obtainable relics and potions, fast enough to give a 10–100×
throughput gain over the C++ search pipeline (~15 s per fight today).

## Scope

In:
- Ironclad cards (75), the colorless cards Ironclad can own, curses, the
  statuses Ironclad's own cards make (Wound, Dazed, Burn).
- Every relic and potion an Ironclad run can have that matters in combat,
  including post-combat heals (they change the outcome value).
- The Champ only, one fixed ascension.

Out:
- Runic Dome.
- Cards that generate cards from another character's pool.
- Other characters' relics and potions.
- Anything that only matters with several enemies or on enemy death
  (Gremlin Horn, The Specimen, Preserved Insect, Sling of Courage), and
  Smoke Bomb (can't escape a boss).

What a single enemy buys:
- No targeting in the action space.
- Cleave equals a single-target hit, and random targets are fixed
  (Sword Boomerang, Juggernaut, Havoc).
- One monster AI, written as arithmetic instead of a per-monster switch.

## Design

- **Cards are table rows:** cost, type, flags, and up to K micro-ops
  `(opcode, a, b)`. Upgraded and unupgraded versions are separate rows.
- **A micro-op VM runs a fixed ring-buffer queue.** It keeps the C++
  addToTop/addToBot semantics and dispatches with `lax.switch`. Under `vmap`,
  every opcode body runs for every lane, so **the cost grows with the number
  of opcodes. Keep the opcode count small and push variety into table
  parameters.**
- **Cards live in fixed slots,** with per-slot `cid`/`up`/`cost`/`misc`. Piles
  are stacks of slot ids plus a length, so draw and discard are O(1) gathers.
- **Relics and powers are counters in `ctr[C]`.** Each is applied at a fixed
  hook point (combat start, turn start/end, card played by type, damage
  dealt, HP lost, block, exhaust, shuffle, potion used, death, combat end).
  The math is masked by relic presence, not branched on.
- **Decision points with a choice are one generic state:** pick k cards from
  a pile, with a filter, then run a follow-up op. This covers Armaments,
  Burning Pact, True Grit+, Headbutt, Warcry, Dual Wield, Exhume, Elixir,
  Gambling Chip, Forethought, Purity, Secret Technique/Weapon, Thinking Ahead
  and Liquid Memories. "Choose 1 of 3 generated cards" (Discovery, the
  Attack/Skill/Power/Colorless potions, Nilry's Codex, Toolbox) is the same
  state over a 3-slot scratch pile.

## Classification

**Plain table rows.** These need only the base primitives: attack, block,
status, draw, energy, lose HP, heal, add card.
- Strike, Defend, Bash, Anger, Cleave, Clothesline, Iron Wave, Pommel Strike,
  Shrug It Off, Thunderclap, Twin Strike, Flex, Inflame, Bloodletting,
  Hemokinesis, Uppercut, Shockwave, Intimidate, Disarm, Sword Boomerang
- Wild Strike, Reckless Charge, Power Through, Immolate, Carnage, Bludgeon,
  Pummel, Seeing Red, Ghostly Armor, Battle Trance, Offering, Impervious,
  Spot Weakness, Dropkick, Entrench, Limit Break, Clash
- Most colorless cards (Bandage Up, Blind, Dark Shackles, Deep Breath,
  Dramatic Entrance, Finesse, Flash of Steel, Good Instincts, Impatience,
  Panacea, Panic Button, Swift Strike, Trip, Bite, J.A.X., Apparition)
- Most potions (Block, Blood, Cultist, Dexterity, Energy, Essence of Steel,
  Explosive, Fear, Fire, Flex, Fruit Juice, Heart of Iron, Liquid Bronze,
  Regen, Speed, Strength, Swift, Weak, Ancient, Fairy)

**Table row plus a formula field.** Damage or block comes from a formula:
strength multiplier, X cost, block, per-tag count, or per-card misc value.
- Heavy Blade, Perfected Strike, Body Slam, Whirlwind, Searing Blow, Rampage,
  Ritual Dagger, Mind Blast, Fiend Fire, Second Wind, Sever Soul, Feed,
  Reaper, Blood for Blood, Transmutation, Chemical X

**Powers as counters and hooks.**
- Demon Form, Metallicize, Combust, Brutality, Berserk, Barricade,
  Corruption, Juggernaut, Rupture, Evolve, Fire Breathing, Feel No Pain,
  Dark Embrace, Rage, Flame Barrier, Double Tap, Sentinel (on exhaust), the
  colorless powers, and every curse (end-of-turn or in-hand triggers)

**Generic mechanisms**, each built once:
- **Card select:** see the list above.
- **Play a card from the draw pile, or replay one.** Uses a small fixed
  card queue: Havoc, Double Tap, Mayhem, Distilled Chaos, Duplication
  Potion, Necronomicon.
- **Random card from a pool table:** Infernal Blade, Jack of All Trades,
  Chrysalis, Metamorphosis, Magnetism, Enchiridion, Dead Branch, Entropic
  Brew.
- **Per-slot cost overrides:** Snecko Eye and Snecko Oil, Madness,
  Enlightenment, Corruption, Mummified Hand, Infernal Blade.

**Relics.** About 75 matter in combat for Ironclad, and all fit
counter-plus-hook:
- Counters with a threshold: Pen Nib, Nunchaku, Kunai, Shuriken, Ornamental
  Fan, Letter Opener, Ink Bottle, Happy Flower, Sundial, Incense Burner,
  Art of War, Velvet-style energy relics, Stone Calendar, Horn Cleat,
  Captain's Wheel, Mercury Hourglass.
- Flat effects at combat start: Anchor, Bag of Marbles, Bag of Preparation,
  Blood Vial, Bronze Scales, Oddly Smooth Stone, Vajra, Lantern, Thread and
  Needle, Philosopher's Stone, Pantograph, Red Mask, Gremlin Visage,
  Clockwork Souvenir, Fossilized Helix, Akabeko, Girya, Du-Vu Doll,
  Mutagenic Strength.
- Damage pipeline: Paper Phrog, The Boot, Torii, Tungsten Rod, Strike Dummy,
  Champion Belt.
- Conditional: Red Skull, Self-Forming Clay, Centennial Puzzle, Runic Cube,
  Magic Flower, Charon's Ashes, Lizard Tail, Unceasing Top, Ice Cream,
  Runic Pyramid, Orichalcum, Calipers, Brimstone, Mark of Pain.
- Random or choice (they use the generic mechanisms): Mummified Hand,
  Warped Tongs, Gambling Chip, Toolbox, Enchiridion, Dead Branch, Nilry's
  Codex, Strange Spoon, Snecko Eye, Bottled Flame/Lightning/Tornado,
  Necronomicon.

**Estimated opcode count for full scope: about 35–45.** Each opcode is a
primitive with table parameters, not one per card.

## Validation

The C++ simulator is the oracle. Planned approach:
1. Export a `BattleContext` into JAX state.
2. Apply one decision on both sides.
3. Compare the resulting states.
4. Cover RNG-sensitive steps with distributional tests.
5. Check fight-level outcome distributions under a fixed policy.

Bit-exact RNG isn't needed for RL.

## Measurements

See README.md.
