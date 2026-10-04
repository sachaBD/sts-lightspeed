// One-core, equal-time starter-deck combat diagnostic, not a win-rate study.
// Compile without PUBLIC_STATE_SEARCH_METRICS against the archived history search
// for the fresh-per-action control. The production search has no implementation switch.
#include "game/GameContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <chrono>
#include <iostream>
#include <memory>

using namespace sts;
using Search = search::PublicBeliefCombatSearch;
using Clock = std::chrono::steady_clock;
namespace {
std::vector<BattleContext> fresh(const BattleContext &observed, int decision) {
    std::vector<BattleContext> states;
    for (int i = 0; i < 8; ++i) {
        auto state = observed;
        const auto seed = static_cast<std::uint64_t>(1000 + decision * 16 + i);
        state.seed = seed;
        state.aiRng = Random(seed + 1); state.cardRandomRng = Random(seed + 2);
        state.miscRng = Random(seed + 3); state.monsterHpRng = Random(seed + 4);
        state.potionRng = Random(seed + 5); state.shuffleRng = Random(seed + 6);
        Search::resampleDraw(state, observed, seed);
        states.push_back(std::move(state));
    }
    return states;
}
}
int main() {
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        GameContext game(CharacterClass::IRONCLAD, seed, 0); game.floorNum = 1;
        BattleContext live; live.init(game, MonsterEncounter::JAW_WORM);
        std::unique_ptr<Search> tree;
        std::int64_t simulations = 0;
        int decisions = 0, reused = 0;
#ifdef PUBLIC_STATE_SEARCH_METRICS
        std::int64_t treeActions = 0;
        int maxDepth = 0;
#endif
        const auto start = Clock::now();
        while (live.outcome == Outcome::UNDECIDED && decisions < 100) {
            const auto decisionStart = Clock::now();
            if (!tree) tree = std::make_unique<Search>(fresh(live, decisions), 42 + decisions, 2);
            tree->maximumActions = 128;
            const auto before = tree->simulations;
            do { tree->search(32); }
            while (Clock::now() - decisionStart < std::chrono::milliseconds(3));
            simulations += tree->simulations - before;
#ifdef PUBLIC_STATE_SEARCH_METRICS
            treeActions += tree->treeActions;
            maxDepth = std::max(maxDepth, tree->maximumTreeDepth);
            reused += tree->retainedVisits > 0;
#endif
            const auto action = Search::mapAction(tree->particles.front(), tree->selectedAction(), live);
            const auto semantic = Search::publicActionKey(live, action);
            action.execute(live);
            ++decisions;
#ifdef PUBLIC_STATE_SEARCH_METRICS
            if (live.outcome == Outcome::UNDECIDED) {
                auto states = tree->advanceParticles(semantic, live);
                if (states.size() != 8) states = fresh(live, decisions);
                tree->rebase(std::move(states), semantic, 42 + decisions);
            }
#else
            tree.reset();
#endif
        }
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "seed=" << seed << " won=" << (live.outcome == Outcome::PLAYER_VICTORY)
                  << " combat_hp=" << live.player.curHp << " decisions=" << decisions
                  << " simulations=" << simulations << " reused_decisions=" << reused
                  << " seconds=" << seconds;
#ifdef PUBLIC_STATE_SEARCH_METRICS
        std::cout << " mean_tree_actions=" << static_cast<double>(treeActions) / simulations
                  << " maximum_tree_depth=" << maxDepth;
#endif
        std::cout << '\n';
        if (live.outcome == Outcome::UNDECIDED) return 1;
    }
}
