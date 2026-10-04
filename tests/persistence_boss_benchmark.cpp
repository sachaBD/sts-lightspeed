// Matched history-tree controls: identical sampler/search; only lifetime differs.
#include "persistence_boss_fixture.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
using namespace sts;
using Search = search::PublicBeliefCombatSearch;
using Clock = std::chrono::steady_clock;
namespace {
std::uint64_t nextSeed(std::uint64_t &value) {
    value += 0x9e3779b97f4a7c15ULL;
    auto v = value;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}
std::vector<BattleContext> fresh(const BattleContext &observed, std::uint64_t seed) {
    std::vector<BattleContext> states;
    states.reserve(8);
    for (int i = 0; i < 8; ++i) {
        auto state = observed;
        auto particleSeed = nextSeed(seed);
        state.seed = nextSeed(particleSeed);
        state.aiRng = Random(nextSeed(particleSeed));
        state.cardRandomRng = Random(nextSeed(particleSeed));
        state.miscRng = Random(nextSeed(particleSeed));
        state.monsterHpRng = Random(nextSeed(particleSeed));
        state.potionRng = Random(nextSeed(particleSeed));
        state.shuffleRng = Random(nextSeed(particleSeed));
        Search::resampleDraw(state, observed, seed);
        states.push_back(std::move(state));
    }
    return states;
}
}
int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "root";
    const int budget = argc > 2 ? std::stoi(argv[2]) : 100;
    const int firstSeed = argc > 3 ? std::stoi(argv[3]) : 2;
    const int seeds = argc > 4 ? std::stoi(argv[4]) : 3;
    const int deck = argc > 5 ? std::stoi(argv[5]) : 0;
    // Diagnostic control of an existing constructor setting, not a runtime switch.
    const bool merge = argc > 6 && std::stoi(argv[6]) != 0;
    if (mode != "root" && mode != "fresh" && mode != "persistent")
        throw std::invalid_argument("mode must be root/fresh/persistent");
    if (budget <= 0 || seeds <= 0 || firstSeed < 0 || deck < 0 || deck > 1)
        throw std::invalid_argument("invalid diagnostic parameters");
    for (auto encounter : {MonsterEncounter::CHAMP, MonsterEncounter::TIME_EATER, MonsterEncounter::AWAKENED_ONE}) {
        for (int seed = firstSeed; seed < firstSeed + seeds; ++seed) {
            auto live = bossFixture(encounter, seed, deck);
            if (mode == "root") {
                std::vector<BattleContext> particles;
                for (int i = 0; i < 8; ++i) {
                    auto state = live;
                    state.aiRng = Random(100 + i); state.shuffleRng = Random(200 + i);
                    state.cardRandomRng = Random(300 + i); state.miscRng = Random(400 + i);
                    Search::resampleDraw(state, live, 500 + i);
                    particles.push_back(std::move(state));
                }
                Search tree(std::move(particles), 42, 2, merge);
                tree.maximumActions = 512;
                const auto start = Clock::now();
                tree.search(budget);
                std::cout << "mode=root boss=" << monsterEncounterEnumNames[static_cast<int>(encounter)]
                          << " seed=" << seed << " deck=" << live.cards.drawPile.size() + live.cards.cardsInHand
                          << " simulations=" << tree.simulations << " nodes=" << tree.nodes.size()
                          << " seconds=" << std::chrono::duration<double>(Clock::now() - start).count()
                          << " chosen_bits=" << tree.selectedAction().bits
                          << " mean_tree_actions=" << double(tree.treeActions) / tree.simulations
                          << " max_tree_depth=" << tree.maximumTreeDepth
                          << " max_tree_turns=" << tree.maximumTreeTurns << std::endl;
                continue;
            }
            std::unique_ptr<Search> tree;
            std::uint64_t previousAction = 0;
            std::int64_t simulations = 0, treeActions = 0, maximumRetained = 0;
            double maintenanceSeconds = 0;
            int decisions = 0, reused = 0, maximumDepth = 0, maximumTurns = 0;
            const auto start = Clock::now();
            while (live.outcome == Outcome::UNDECIDED && decisions < 240) {
                const auto decisionStart = Clock::now();
                const auto publicSeed = Search::publicObservation(live);
                auto states = fresh(live, publicSeed);
                if (!tree) tree = std::make_unique<Search>(std::move(states), publicSeed, 2, merge);
                else tree->rebase(std::move(states), previousAction, publicSeed);
                tree->maximumActions = 512;
                maintenanceSeconds += std::chrono::duration<double>(Clock::now() - decisionStart).count();
                reused += tree->retainedVisits > 0;
                maximumRetained = std::max(maximumRetained, tree->retainedVisits);
                do { tree->search(32); }
                while (Clock::now() - decisionStart < std::chrono::milliseconds(budget));
                simulations += tree->simulations;
                treeActions += tree->treeActions;
                maximumDepth = std::max(maximumDepth, tree->maximumTreeDepth);
                maximumTurns = std::max(maximumTurns, tree->maximumTreeTurns);
                const auto action = Search::mapAction(tree->particles.front(), tree->selectedAction(), live);
                previousAction = tree->actionKey(live, action);
                action.execute(live);
                ++decisions;
                if (mode == "fresh") tree.reset();
            }
            std::cout << "mode=" << mode << " boss=" << monsterEncounterEnumNames[static_cast<int>(encounter)]
                      << " seed=" << seed << " deck=" << deck << " merge=" << merge
                      << " won=" << (live.outcome == Outcome::PLAYER_VICTORY)
                      << " combat_hp=" << live.player.curHp << " decisions=" << decisions
                      << " simulations=" << simulations << " mean_tree_actions=" << double(treeActions) / simulations
                      << " max_tree_depth=" << maximumDepth << " max_tree_turns=" << maximumTurns
                      << " reused_decisions=" << reused << " max_retained_visits=" << maximumRetained
                      << " maintenance_seconds=" << maintenanceSeconds
                      << " seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << std::endl;
            if (live.outcome == Outcome::UNDECIDED) throw std::runtime_error("censored boss benchmark");
        }
    }
}
