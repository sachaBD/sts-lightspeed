#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error{message}; }

sts::BattleContext fixture(sts::CardSelectTask prior) {
    sts::BattleContext state;
    state.inputState = sts::InputState::PLAYER_NORMAL;
    state.player.cc = sts::CharacterClass::IRONCLAD;
    state.player.energy = 0;
    state.cardSelectInfo.cardSelectTask = prior;
    state.monsters.monsterCount = state.monsters.monstersAlive = 1;
    state.monsters.arr[0].id = sts::MonsterId::THE_CHAMP;
    state.monsters.arr[0].curHp = state.monsters.arr[0].maxHp = 440;
    state.cards.discardPile.push_back(sts::CardInstance{sts::CardId::BASH});
    state.cards.drawPile.push_back(sts::CardInstance{sts::CardId::DEFEND_RED});
    state.potionCapacity = 2; state.potionCount = 1;
    state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
    state.potions[0] = sts::Potion::LIQUID_MEMORIES;
    return state;
}

int main() {
    try {
        sts::CardSelectInfo defaults;
        require(defaults.cardSelectTask == sts::CardSelectTask::INVALID, "default task not initialized");
        // Every prior menu task, including no prior menu: one discard must always return at zero cost.
        for (int prior = 0; prior <= int(sts::CardSelectTask::WISH); ++prior) {
            auto state = fixture(static_cast<sts::CardSelectTask>(prior));
            const sts::search::Action potion{sts::search::ActionType::POTION, 0, 0};
            require(potion.isValidAction(state), "fixture potion illegal");
            potion.execute(state);
            require(state.inputState == sts::InputState::PLAYER_NORMAL && state.cards.cardsInHand == 1,
                    "one-card potion opened a menu");
            require(state.cards.discardPile.empty() && state.cards.hand[0].costForTurn == 0,
                    "Liquid Memories fast path depends on prior task");
            require(sts::search::Action{sts::search::ActionType::CARD, 0, 0}.isValidAction(state),
                    "returned card is not playable at zero energy");
        }
        auto hologram = fixture(sts::CardSelectTask::LIQUID_MEMORIES_POTION);
        hologram.addToBot(sts::Actions::BetterDiscardPileToHandAction(1, sts::CardSelectTask::HOLOGRAM));
        hologram.inputState = sts::InputState::EXECUTING_ACTIONS;
        hologram.executeActions();
        require(hologram.cards.hand[0].costForTurn == 2, "non-free recovery inherited old Liquid Memories task");
        std::cout << "Liquid Memories fast-path and stale-task regressions passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
