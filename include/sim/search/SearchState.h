#ifndef STS_SEARCH_STATE_H
#define STS_SEARCH_STATE_H

#include "sim/search/Action.h"
#include <cstdint>
#include <vector>

namespace sts::search {

// Whitelisted enemy memory derived from visible intents/actions, not hidden AI data.
std::uint64_t publicMonsterHistory(const Monster &monster);

// Ordered vectors remain in the simulator; search identity uses multisets.
bool isSettledSearchState(const BattleContext &state);
// Practical public decision-state abstraction, not a complete hidden belief.
std::vector<std::uint64_t> publicSearchWords(const BattleContext &state);
void publicSearchWords(const BattleContext &state, std::vector<std::uint64_t> &out);
std::uint64_t public_search_key(const BattleContext &state);
std::uint64_t hashSearchWords(const std::vector<std::uint64_t> &words);
std::uint64_t semantic_action_key(const BattleContext &state, Action action);
Action mapSemanticAction(const BattleContext &source, Action action, const BattleContext &target);

}
#endif
