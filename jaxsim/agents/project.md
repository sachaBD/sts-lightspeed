# Project: goal, scope and how the user works

## Goal
An AlphaZero-style training loop for **Ironclad vs The Champ** (act 2 boss) that is **reasonably fast and
fully JAX**: simulator, mctx search and network in one compiled program on **one GPU** (RTX 3060 12 GB,
possibly a 3090).
- The previous pipeline is C++ search with 8 particles × 1k sims, at about 15 s per fight.
- The near-term milestone is **playable**: real human_champ decks run end to end (#24).

## Fixed decisions (agreed with the user, don't reopen)
- **A20 only.** Win-only reward (truncation is not a loss; the consumer scores it).
- **Any deck.** The network takes the deck as input (per-slot tokens). Training decks come from the
  human_champ library in sts-ml. The user picks decks around what jaxsim supports, not the other way round.
- **Sim/play split:**
  - jaxsim lives in sts-lightspeed (`jaxsim/`, an installable uv package);
  - the play side is in sts-ml (`agents/combat/az`, `apps/az_train`, `apps/az_bench`);
  - the boundary is `jaxsim.Env` (pure functions).
- **Pure functions.** `step(state, action, key)`, no RNG in the state, particles = different keys.
- **Count-based identity slots** (D=48), slot-aligned actions (A=55), sequential picks for choices, integer
  observations, potions as identity slots too. 2 known-top cards. New network weights (no PV port).
- **The C++ simulator is the reference.** When it differs from the real game, fix it in **both** C++ and jaxsim,
  in a commit with a clear message (done for Iron Wave, Disarm+, Seeing Red, Rage, card colours).
- **Out of scope:** bottom-of-draw effects (Forethought), Frozen Eye, Bottled cards, Runic Dome,
  multi-enemy-only relics, Smoke Bomb, other characters' content.

## How the user likes to work
- **Concise status and sync-ups.** When asked to sync, just talk: no tool calls, no code dumps.
- **Track work in GitHub issues.** `AGENDA.md` is a simple summary. Make an issue for anything not done or not bulletproof, and move on.
- **Permission to merge PRs and use git freely.** Only stop for a true decision.
  - Pattern: worktree in `.worktrees/<name>` → commits → PR → `gh pr merge --rebase --delete-branch`
    → remove the worktree.
  - sts-ml: commit straight to `main`.
- **Focus on functionality towards playable**, not wall-clock benchmarks. CPU numbers are only a rough guide;
  the target is the GPU. Don't benchmark after every change. A slowdown is fine if it buys needed function.
- **Tests: 80/20.** A fast default suite (about 30 s). Full robustness comes after playable.
- **Simple over clever** when something is a time sink: pick the simple correct option, file an issue for the
  better one (e.g. Evolve rounds, #16).
- The user sometimes plays Slay the Spire on the same Mac, which loads the CPU and makes timings noisy.
