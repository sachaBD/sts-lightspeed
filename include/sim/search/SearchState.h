#ifndef STS_SEARCH_STATE_H
#define STS_SEARCH_STATE_H

#include "sim/search/Action.h"
#include <cstdint>
#include <vector>

namespace sts::search {

// Canonical internal representation at settled player decisions. Pending card-selection
// queues retain their original indexing. Draw order is never changed.
void canonicalizeSearchState(BattleContext &state);
bool isSettledSearchState(const BattleContext &state);

// Public projection only, NOT a sufficient belief/transposition key.
std::vector<std::uint64_t> publicSearchWords(const BattleContext &state);
std::uint64_t public_search_key(const BattleContext &state);

// Complete future-relevant state at canonical, settled decisions, including hidden state and RNGs.
// Never use this per-particle fingerprint to select a public policy. Fusion compares
// entire observation-conditioned particle cohorts (including multiplicities).
std::vector<std::uint64_t> searchStateWords(const BattleContext &state);
std::uint64_t hashSearchWords(const std::vector<std::uint64_t> &words);
std::uint64_t semantic_action_key(const BattleContext &state, Action action);
Action mapSemanticAction(const BattleContext &source, Action action, const BattleContext &target);

}
#endif
