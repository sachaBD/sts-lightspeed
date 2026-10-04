// Fixed-work boss/deck stress diagnostic. Also builds against the archived history
// search for a control; no production implementation switches are introduced.
#include "search_boss_fixture.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
using namespace sts;

int main(int argc, char **argv) {
    const int budget = argc > 1 ? std::stoi(argv[1]) : 3000;
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
    if (budget <= 0 || repeats <= 0) throw std::invalid_argument("positive benchmark budget required");
    for (auto encounter : {MonsterEncounter::CHAMP, MonsterEncounter::TIME_EATER, MonsterEncounter::AWAKENED_ONE}) {
        auto root = bossFixture(encounter, 7);
        std::vector<BattleContext> particles;
        for (int i = 0; i < 8; ++i) {
            auto state = root;
            state.aiRng = Random(100 + i); state.shuffleRng = Random(200 + i);
            state.cardRandomRng = Random(300 + i); state.miscRng = Random(400 + i);
            search::PublicBeliefCombatSearch::resampleDraw(state, root, 500 + i);
            particles.push_back(std::move(state));
        }
        for (int repeat = 0; repeat < repeats; ++repeat) {
            search::PublicBeliefCombatSearch tree(particles, 42, 2);
            tree.maximumActions = 512;
            const auto start = std::chrono::steady_clock::now();
            tree.search(budget);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "boss=" << monsterEncounterEnumNames[static_cast<int>(encounter)]
                      << " deck=32" << " repeat=" << repeat
                      << " simulations=" << tree.simulations << " nodes=" << tree.nodes.size()
                      << " seconds=" << seconds << " chosen_bits=" << tree.selectedAction().bits;
#ifdef PUBLIC_STATE_SEARCH_METRICS
            std::size_t expanded = 0, bytes = 0;
            for (const auto &[key, node] : tree.nodes) {
                expanded += node->expanded; bytes += node->signature.size() * sizeof(std::uint64_t);
            }
            std::vector<std::pair<std::uint64_t, std::uint64_t>> evidence;
            for (const auto &edge : tree.root().edges) {
                std::uint64_t valueBits;
                std::memcpy(&valueBits, &edge.valueSum, sizeof(valueBits));
                evidence.emplace_back(edge.semanticKey, valueBits ^ static_cast<std::uint64_t>(edge.visits));
            }
            std::sort(evidence.begin(), evidence.end());
            std::uint64_t evidenceHash = 0;
            for (const auto &[key, value] : evidence) evidenceHash = (evidenceHash ^ key ^ value) * 0x100000001b3ULL;
            std::cout << " root_evidence_hash=" << evidenceHash << " expanded=" << expanded << " signature_bytes=" << bytes
                      << " hits=" << tree.transpositionHits
                      << " mean_tree_actions=" << static_cast<double>(tree.treeActions) / tree.simulations
                      << " max_tree_depth=" << tree.maximumTreeDepth << " max_tree_turns=" << tree.maximumTreeTurns;
#endif
            std::cout << std::endl;
        }
    }
}
