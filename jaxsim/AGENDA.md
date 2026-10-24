# Agenda

Goal: **playable**. The agent plays real human_champ decks end to end in jaxsim self-play on the GPU.
Tracking issue: [#24](https://github.com/sachaBD/sts-lightspeed/issues/24).

| Area | Issue | What |
|---|---|---|
| Content | ~~#17~~ ~~#19~~ ~~#6~~ ~~#20~~ | Done: choice cards, curses + colorless, 62 relics, 21 potions |
| | [#18](https://github.com/sachaBD/sts-lightspeed/issues/18) | Havoc, Double Tap, Infernal Blade (completes the 75 Ironclad cards) |
| | [#29](https://github.com/sachaBD/sts-lightspeed/issues/29) | Remaining relics (random/choice ones, Akabeko, Runic Cube, ...) |
| | [#31](https://github.com/sachaBD/sts-lightspeed/issues/31) | Remaining potions (generated-card choices, multi-picks, replays) |
| Play side | [sts-ml#11](https://github.com/sachaBD/sts-ml/issues/11) | human_champ starts → jaxsim, plus a report of what blocks which decks |
| | [sts-ml#12](https://github.com/sachaBD/sts-ml/issues/12) | First GPU run (RTX 3060) |
| | [sts-ml#10](https://github.com/sachaBD/sts-ml/issues/10) | Self-play training on real decks |
| Tests (80/20) | [#21](https://github.com/sachaBD/sts-lightspeed/issues/21) | Distribution checks for the real random path |

After playable:
- [#22](https://github.com/sachaBD/sts-lightspeed/issues/22) robust testing;
- [#23](https://github.com/sachaBD/sts-lightspeed/issues/23) performance pass on the GPU.

## Flow

```
sts-ml#11 support report ──► sets the order within #18, #29, #31
#18 ─► #29 / #31 as decks need them       #21 alongside
                                  └──► sts-ml#12 GPU ─► sts-ml#10 training on real decks
```
