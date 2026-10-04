// Small equal-time whole-boss diagnostic, not a win-rate qualification.
// Both arms preserve identical public draw knowledge. The history control starts
// fresh each action; the production arm propagates particles and advances its DAG.
#include "search_boss_fixture.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <tuple>
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
        // Use the same sampler for both arms, including Headbutt knowledge.
        auto &cards = state.cards.drawPile;
        auto first = cards.begin() + state.cards.knownDrawBottom;
        auto last = cards.end() - state.cards.knownDrawTop;
        std::sort(first, last, [](const auto &a, const auto &b) {
            return std::tie(a.id,a.upgraded,a.specialData,a.cost,a.costForTurn,a.freeToPlayOnce,a.retain)
                < std::tie(b.id,b.upgraded,b.specialData,b.cost,b.costForTurn,b.freeToPlayOnce,b.retain);
        });
        std::default_random_engine random(seed);
        std::shuffle(first, last, random);
        states.push_back(std::move(state));
    }
    return states;
}
}
int main(int argc, char **argv) {
    const int fixedBudget = argc > 1 ? std::stoi(argv[1]) : 0;
    const std::string onlyBoss = argc > 2 ? argv[2] : "";
    if (fixedBudget < 0) throw std::invalid_argument("negative benchmark budget");
    for (auto encounter : {MonsterEncounter::CHAMP, MonsterEncounter::TIME_EATER, MonsterEncounter::AWAKENED_ONE}) {
        if (!onlyBoss.empty() && onlyBoss != monsterEncounterEnumNames[static_cast<int>(encounter)]) continue;
        for (std::uint64_t seed = 0; seed < 2; ++seed) {
            auto live = bossFixture(encounter, seed);
            std::unique_ptr<Search> tree;
            std::int64_t simulations = 0, maximumRetained = 0;
            int decisions = 0, reused = 0;
            const auto start = Clock::now();
            while (live.outcome == Outcome::UNDECIDED && decisions < 240) {
                const auto decisionStart = Clock::now();
                if (!tree) tree = std::make_unique<Search>(fresh(live, decisions), 42 + decisions, 2);
                tree->maximumActions = 512;
                const auto before = tree->simulations;
                if (fixedBudget) tree->search(fixedBudget);
                else {
                    do { tree->search(32); }
                    while (Clock::now() - decisionStart < std::chrono::milliseconds(100));
                }
                simulations += tree->simulations - before;
#ifdef PUBLIC_STATE_SEARCH_METRICS
                reused += tree->retainedVisits > 0;
                maximumRetained = std::max(maximumRetained, tree->retainedVisits);
#endif
                const auto action = Search::mapAction(tree->particles.front(), tree->selectedAction(), live);
                const auto semantic = Search::publicActionKey(live, action);
                action.execute(live);
                ++decisions;
#if defined(PUBLIC_STATE_SEARCH_METRICS) && !defined(PUBLIC_SEARCH_FRESH_CONTROL)
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
            std::cout << "boss=" << monsterEncounterEnumNames[static_cast<int>(encounter)]
                      << " seed=" << seed << " won=" << (live.outcome == Outcome::PLAYER_VICTORY)
                      << " combat_hp=" << live.player.curHp << " decisions=" << decisions
                      << " simulations=" << simulations << " reused_decisions=" << reused
                      << " max_retained_visits=" << maximumRetained << " seconds=" << seconds << std::endl;
            if (live.outcome == Outcome::UNDECIDED) throw std::runtime_error("censored boss benchmark");
        }
    }
}
