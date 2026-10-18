# Agenda

Goal: **playable**. The agent plays real human_champ decks end to end in jaxsim self-play on the GPU.
Tracking issue: [#24](https://github.com/sachaBD/sts-lightspeed/issues/24).

| Area | Issue | What |
|---|---|---|
| Content | [#17](https://github.com/sachaBD/sts-lightspeed/issues/17) | Choice cards: Armaments, Headbutt, True Grit, Warcry, Burning Pact, Dual Wield, Exhume |
| | [#18](https://github.com/sachaBD/sts-lightspeed/issues/18) | Havoc, Double Tap, Infernal Blade (completes the 75 Ironclad cards) |
| | [#19](https://github.com/sachaBD/sts-lightspeed/issues/19) | Curses and colorless cards in real decks |
| | [#6](https://github.com/sachaBD/sts-lightspeed/issues/6) | Relics (the ones in human decks first) |
| | [#20](https://github.com/sachaBD/sts-lightspeed/issues/20) | Potions (or strip them from starts at first) |
| Play side | [sts-ml#11](https://github.com/sachaBD/sts-ml/issues/11) | human_champ starts → jaxsim, plus a report of what blocks which decks |
| | [sts-ml#12](https://github.com/sachaBD/sts-ml/issues/12) | First GPU run (RTX 3060) |
| | [sts-ml#10](https://github.com/sachaBD/sts-ml/issues/10) | Self-play training on real decks |
| Tests (80/20) | [#21](https://github.com/sachaBD/sts-lightspeed/issues/21) | Distribution checks for the real random path |

After playable:
- [#22](https://github.com/sachaBD/sts-lightspeed/issues/22) robust testing;
- [#23](https://github.com/sachaBD/sts-lightspeed/issues/23) performance pass on the GPU;
- [#16](https://github.com/sachaBD/sts-lightspeed/issues/16) single-sample Evolve chain.

## Flow

```
sts-ml#11 support report ──► sets the order within #19 and #6
#17 ─► #18 ─► #19 ─► #6 ─► (#20)          #21 alongside
                                  └──► sts-ml#12 GPU ─► sts-ml#10 training on real decks
```
