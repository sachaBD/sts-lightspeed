# Agent knowledge library (read this first when resuming)

Macro-level knowledge from building jaxsim, for an agent starting with fresh context. These notes record the
*why* and the traps. The *what* lives in the code, in `../DESIGN.md` (design and C++ rules copied) and in
`../AGENDA.md` (current plan; issues are the source of truth).

| File | Read when |
|---|---|
| [project.md](project.md) | Always. Goal, scope, the user's preferences and how they like to work |
| [architecture.md](architecture.md) | Before touching code: repos, boundary, state model, step pipeline, file map |
| [cpp-semantics.md](cpp-semantics.md) | Before adding any card, relic or potion: C++ behaviour that is easy to get wrong |
| [workflow.md](workflow.md) | Setting up, building the oracle, testing, adding content, debugging parity |
| [lessons.md](lessons.md) | Before deciding anything non-trivial: mistakes already made and why |

Resume checklist:
1. `gh issue view 24 --repo sachaBD/sts-lightspeed` (milestone tracker) and `../AGENDA.md`.
2. `git log --oneline -15` in sts-lightspeed and sts-ml.
3. Build the oracle and run `uv run pytest` in `jaxsim/` (about 30 s). It must be green before you change anything.
4. Keep these notes current: when you learn a non-obvious rule or make a decision with the user, add it here.
