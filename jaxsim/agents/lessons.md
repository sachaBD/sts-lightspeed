# Lessons (mistakes made, and what to do instead)

- **Don't invent training curricula to force learning.** With 12 cards, win-only reward and an A20 Champ,
  random play never wins. I started tuning HP up to 1000, strong decks and an END_TURN bias. The user stopped
  it: learning waits for real decks; throughput and correctness are what to show until then.
- **Don't over-index on wall-clock benchmarks.** Adding content makes steps slower, and that's expected. CPU
  timings are noisy and only a rough guide to the GPU. Example: the vectorised draw looked 2× slower on CPU at
  first, but it is the right GPU design. Benchmark rarely, compare back-to-back, and say so when load is high.
- **The batch-3 slowdown (−62% simulator) came from unrolled loops**: 2 draw points × 10 single-card
  iterations, all paid by every lane under vmap. Replacing loops with one vectorised op is the usual fix.
  Avoid unrolled per-card loops.
- **Reason about bounds from the game, not guesses.** Evolve chains are bounded by the 10-card hand (≤ 9
  rounds, worst case unupgraded Evolve). I first used 2 or 3 rounds, which was wrong in rare chains.
- **Bump the contract when a consumer exists.** Widening `Obs.player` broke sts-ml's network silently, and the
  benchmark found it. Now the contract is pinned in a test.
- **Fixed-size state fields need explicit bounds plus an error bit**, never silent clamping (ERR_SLOT_OVERFLOW,
  ERR_QUEUE_OVERFLOW).
- **Full-state `tree_where` selects are expensive under vmap.** Prefer field-level `jnp.where` or `want`
  masks (as in `find_or_alloc`).
- **Parity failures usually point at real semantics.** Trust the C++, read the exact function, and fix the
  rule. If the real game disagrees with the C++, fix both. Harness gaps (exhausted-from-hand, created-in-hand,
  identity cost changes, forced draw order) also showed up as "mismatches", so check the harness logic too.
- **Long autonomous stretches lose the user.** Give short status summaries at milestones and when asked;
  explain changes in plain terms (what changed, why, what next) before deep detail.
