# Workflow

## Setup (Mac)
- Python: `uv` (installed). `jaxsim/.python-version` = 3.12. Run `uv sync` in `jaxsim/`.
- cmake/ninja: `uv tool install cmake ninja`, with `~/.local/bin` on PATH.
- **C++ link fails with the default 27.0 SDK.** Use
  `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.
- Oracle: `SDKROOT=... tools/build_oracle.sh` → `oracle/build/jaxsim_oracle`. **Rebuild it after any C++
  change**, and in every new worktree (parity is skipped without it, and the skip is silent).
- sts-ml: `STS_LIGHTSPEED_DIR=../sts-lightspeed uv pip install --python .venv/bin/python -r requirements-jax.txt`.
  Reinstall jaxsim (`uv pip install -e <path>/jaxsim`) when switching worktrees.
- `apps/common/launch.sh` needed a macOS bash 3.2 fix for empty arrays (done).

## Tests
- `uv run pytest`: about 30 s. Contract, data checks, targeted card tests, 60 parity fights.
- `PARITY_FIGHTS=300 uv run pytest`: the full parity run, before big merges.
- `uv run python -m tests.oracle.coverage`: plays per card, outcomes, Champ moves (C++ side). Use it to check
  new cards are actually exercised.
- sts-ml: `PYTHONPATH=. .venv/bin/python -m agents.combat.az.test_az`.

## Adding a card (checklist)
1. Read its C++ `case CardId::X` and every action or status it touches (see cpp-semantics.md).
2. Add a row to `_core/tables/cards.py`. New behaviour goes in a column or a `special`, applied in
   `cards.play_pre` / `play_post`. Hook-like behaviour goes in `hooks.py` or `step.end_*`.
3. Cost, type, exhaust and ethereal are checked against C++ automatically
   (`test_card_table_matches_cpp`). A mismatch is either your typo or a C++ bug; check the real game.
4. Run parity. Mismatches print the slot table and the C++ player state.
5. Add a targeted test in `tests/test_cards.py` for any rare rule random play won't reach.
6. Update `DESIGN.md` (supported list, any C++ behaviour copied) and these notes if you learned something.

## How the parity harness works (`tests/test_parity.py`, `tests/oracle/`)
- Random decks (3 kinds: uniform random play; card-first play at 300 HP; power-heavy at 250 HP). Random legal
  C++ actions are mapped to jaxsim slots. Every step: state fields + legal-action set compared.
- JAX runs the **forced path**. The oracle reports `drawn` and `exhausted` per request, tracked by card uid
  (a card counts as drawn if it is in hand now and was in the draw/discard pile before, or it is the played or
  limbo card or a Warcry pick that came back; after END_TURN every pre-existing card in hand). The Champ roll
  comes from replaying a copy of `aiRng`.
- Forced draws: known-top cards first (deterministic, no forced entry needed), then statuses first (Evolve), or
  non-statuses first for the end-turn Dark Embrace draw (Fire Breathing timing). Forced counts match by card
  group (card, upgrade, misc), so identity changes mid-step (Blood for Blood cost) don't break them.
- `PARITY_DEBUG=1 uv run pytest -s ...` prints each select, the C++ hand/top of draw/enemy after each step, and
  the pre-step JAX state.
- Card identity mapping:
  - uses (card, upgraded, cost, misc); misc only for Rampage;
  - for the drawn-card computation, cost is ignored (it can change mid-step);
  - new identities map to the next never-used slot.
- Once the fight is over only done/won/hp/max hp are compared.
- Debugging: run one fight, `uv run pytest -q "tests/test_parity.py::test_random_fight[N]"`, and read the
  diff + slot table. Most failures so far were (a) a C++ ordering rule, (b) a harness bookkeeping gap, or
  (c) a C++ bug.

## Benchmark (sts-ml `apps/az_bench`)
- About 40 s in quick mode, with compile cached in `build/jax-cache`; `--full` for more.
- Only for occasional checks; numbers are recorded in `agents/combat/az/README.md`.
