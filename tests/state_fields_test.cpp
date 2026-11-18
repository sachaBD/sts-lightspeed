#include "combat/BattleContext.h"
#include "combat/StateFields.h"
#include "constants/CardPools.h"
#include "game/GameContext.h"
#include "sim/search/Action.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace sts;
using search::Action;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error{message};
}

// Every leaf of a state's walk, as words.
std::vector<std::uint64_t> leaves(const BattleContext &state) {
    std::vector<std::uint64_t> out;
    forEachField(state, [&out](const auto &leaf) {
        using T = std::decay_t<decltype(leaf)>;
        if constexpr (std::is_same_v<T, Random>) {
            out.insert(out.end(), {std::uint64_t(leaf.counter), leaf.seed0, leaf.seed1});
        } else {
            out.push_back(static_cast<std::uint64_t>(leaf));
        }
    });
    return out;
}

// A fresh state filled from those words.
BattleContext filled(const std::vector<std::uint64_t> &words) {
    BattleContext state;
    std::size_t next = 0;
    forEachField(state, [&](auto &leaf) {
        using T = std::decay_t<decltype(leaf)>;
        require(next < words.size(), "fill ran past the walk it was given");
        if constexpr (std::is_same_v<T, Random>) {
            leaf.counter = static_cast<std::int32_t>(words.at(next++));
            leaf.seed0 = words.at(next++);
            leaf.seed1 = words.at(next++);
        } else {
            leaf = static_cast<T>(words[next++]);
        }
    });
    require(next == words.size(), "fill must use the whole walk");
    return state;
}

Action randomAction(const BattleContext &state, std::mt19937_64 &random) {
    const auto actions = Action::getAllActionsInState(state);
    require(!actions.empty(), "an undecided fight has a legal action");
    return actions[random() % actions.size()];
}

// The walk names everything play depends on: a copy filled from it plays on exactly like the original.
void copiesPlayAlike() {
    const MonsterEncounter encounters[] = {
        MonsterEncounter::JAW_WORM, MonsterEncounter::GREMLIN_NOB, MonsterEncounter::THREE_SENTRIES,
        MonsterEncounter::SLIME_BOSS, MonsterEncounter::GREMLIN_LEADER, MonsterEncounter::CHAMP,
        MonsterEncounter::AUTOMATON, MonsterEncounter::COLLECTOR, MonsterEncounter::REPTOMANCER,
        MonsterEncounter::AWAKENED_ONE, MonsterEncounter::TIME_EATER, MonsterEncounter::DONU_AND_DECA};
    std::mt19937_64 random{11};
    int copies = 0;
    for (std::uint64_t seed = 1; seed <= 48; ++seed) {
        GameContext game{CharacterClass::IRONCLAD, seed, 20};
        game.floorNum = 30;
        for (int i = 0; i < 15; ++i) {  // a varied deck, so play reaches many kinds of state
            Card card{CombatCardPool::getCardAt(CharacterClass::IRONCLAD, static_cast<int>(random() % 70))};
            card.upgraded = random() % 2;
            game.deck.obtainRaw(card);
        }
        BattleContext state;
        state.init(game, encounters[seed % std::size(encounters)]);
        while (state.outcome == Outcome::UNDECIDED && state.turn < 20) {
            if (state.actionQueue.size == 0) {  // the queue holds code, so only a state without one is all data
                auto copy = filled(leaves(state));
                require(leaves(copy) == leaves(state), "a filled state walks like its source");
                ++copies;
                auto original = state;
                std::mt19937_64 ahead{random()};
                for (int step = 0; step < 30 && original.outcome == Outcome::UNDECIDED; ++step) {
                    const auto action = randomAction(original, ahead);
                    action.execute(original);
                    action.execute(copy);
                    require(leaves(copy) == leaves(original), "a filled state plays on like its source");
                }
            }
            randomAction(state, random).execute(state);
        }
    }
    require(copies > 200, "the fights must reach many states");
}
}  // namespace

int main() {
    try {
        copiesPlayAlike();
        std::cout << "state fields tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "state fields test failed: " << e.what() << '\n';
        return 1;
    }
}
