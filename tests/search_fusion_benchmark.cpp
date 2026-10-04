// Also compile this file against the original PBCS header/source for a matched control.
#include "game/GameContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <chrono>
#include <iostream>
using namespace sts;
int main() {
    GameContext game(CharacterClass::IRONCLAD, 0, 0);
    game.floorNum = 1;
    BattleContext root; root.init(game, MonsterEncounter::JAW_WORM);
    std::vector<BattleContext> particles;
    for (int i = 0; i < 8; ++i) {
        auto state = root;
        search::PublicBeliefCombatSearch::resampleDraw(state, root, 100 + i);
        particles.push_back(std::move(state));
    }
    for (int repeat = 0; repeat < 5; ++repeat) {
        search::PublicBeliefCombatSearch tree(particles, 42, 2);
        tree.maximumActions = 3;
        const auto start = std::chrono::steady_clock::now();
        while (tree.simulations < 25000) {
            const auto ids = tree.requestBatch(1, 25000, 0, 0);
            for (const auto id : ids) tree.submit(id, .5);
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "bounded repeat=" << repeat << " simulations=" << tree.simulations
                  << " nodes=" << tree.nodes.size() << " seconds=" << seconds;
#ifdef PUBLIC_STATE_SEARCH_METRICS
        std::size_t expanded = 0, signatureBytes = 0;
        for (const auto &[key, node] : tree.nodes) {
            expanded += node->expanded; signatureBytes += node->signature.size() * sizeof(std::uint64_t);
        }
        std::cout << " expanded_nodes=" << expanded << " signature_bytes=" << signatureBytes
                  << " mean_tree_actions=" << static_cast<double>(tree.treeActions) / tree.simulations
                  << " maximum_tree_depth=" << tree.maximumTreeDepth
                  << " maximum_tree_turns=" << tree.maximumTreeTurns;
#endif
        std::cout << '\n';
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        search::PublicBeliefCombatSearch tree(particles, 42, 2);
        tree.maximumActions = 128;
        const auto start = std::chrono::steady_clock::now();
        tree.search(2000);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "guided repeat=" << repeat << " simulations=" << tree.simulations
                  << " nodes=" << tree.nodes.size() << " seconds=" << seconds;
#ifdef PUBLIC_STATE_SEARCH_METRICS
        std::size_t expanded = 0, signatureBytes = 0;
        for (const auto &[key, node] : tree.nodes) {
            expanded += node->expanded; signatureBytes += node->signature.size() * sizeof(std::uint64_t);
        }
        std::cout << " expanded_nodes=" << expanded << " signature_bytes=" << signatureBytes
                  << " mean_tree_actions=" << static_cast<double>(tree.treeActions) / tree.simulations
                  << " maximum_tree_depth=" << tree.maximumTreeDepth
                  << " maximum_tree_turns=" << tree.maximumTreeTurns;
#endif
        std::cout << '\n';
    }
}
