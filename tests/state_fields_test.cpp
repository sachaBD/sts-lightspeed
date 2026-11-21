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

BattleContext fixture(std::initializer_list<CardId> hand) {
    GameContext game{CharacterClass::IRONCLAD, 17, 20};
    BattleContext state;
    state.init(game, MonsterEncounter::JAW_WORM);
    state.cards = CardManager{};
    state.resolution.clear();
    state.monsters.arr[0].curHp = state.monsters.arr[0].maxHp = 1000;
    for (auto id : hand) state.cards.createTempCardInHand(CardInstance{id});
    return state;
}

void play(BattleContext& state, int slot = 0) {
    const Action action{search::ActionType::CARD, slot, 0};
    require(action.isValidAction(state), "fixture card must be legal");
    action.execute(state);
}

void pickAndCompare(BattleContext& state) {
    auto copy = filled(leaves(state));
    require(leaves(copy) == leaves(state), "suspended state did not round trip");
    const auto action = Action::getAllActionsInState(state).front();
    action.execute(state);
    action.execute(copy);
    require(leaves(copy) == leaves(state), "restored choice resumed differently");
}

void cardChoices() {
    auto state = fixture({CardId::BURNING_PACT, CardId::DEFEND_RED, CardId::STRIKE_RED});
    for (int i = 0; i < 8; ++i) state.cards.createTempCardInDrawPile(0, CardInstance{CardId::BASH});
    play(state);
    require(state.inputState == InputState::CARD_SELECT && state.cards.drawPile.size() == 8,
            "Burning Pact must wait for its exhaust choice before drawing");
    pickAndCompare(state);
    require(state.inputState == InputState::PLAYER_NORMAL && state.cards.drawPile.size() == 6,
            "Burning Pact must resume its draws and finish");
    require(state.cards.discardPile.size() == 1 && state.resolution.empty(), "card cleanup did not finish");
}

void nestedAutoplay() {
    auto state = fixture({CardId::HAVOC, CardId::STRIKE_RED});
    state.cards.createTempCardInDrawPile(0, CardInstance{CardId::HEADBUTT});
    state.cards.createTempCardInDiscard(CardInstance{CardId::DEFEND_RED});
    state.cards.createTempCardInDiscard(CardInstance{CardId::BASH});
    play(state);
    require(state.inputState == InputState::CARD_SELECT && state.cardSelectInfo.cardSelectTask == CardSelectTask::HEADBUTT,
            "autoplay must pause at the nested Headbutt choice");
    require(state.cards.discardPile.size() == 2, "parent Havoc must remain in play during its child");
    pickAndCompare(state);
    require(state.inputState == InputState::PLAYER_NORMAL && state.resolution.empty(), "parent did not resume");
    require(state.cards.exhaustPile.size() == 1 && state.cards.exhaustPile[0].id == CardId::HEADBUTT,
            "Havoc must exhaust its played card after the choice");
}

void exhaustBatch() {
    auto state = fixture({CardId::FIEND_FIRE, CardId::WOUND, CardId::DEFEND_RED});
    state.player.buff<PS::DARK_EMBRACE>(1);
    for (int i = 0; i < 10; ++i) state.cards.createTempCardInDrawPile(0, CardInstance{CardId::STRIKE_RED});
    play(state);
    require(state.cards.exhaustPile.size() == 3, "Fiend Fire must exhaust its original hand and itself");
    require(state.cards.cardsInHand == 3, "exhaust triggers must not exhaust the newly drawn cards");
    for (int i = 0; i < state.cards.cardsInHand; ++i)
        require(state.cards.hand[i].id == CardId::STRIKE_RED, "unexpected card remained after Fiend Fire");
}

void turnChoices() {
    auto state = fixture({CardId::STRIKE_RED});
    state.player.buff<PS::TOOLS_OF_THE_TRADE>(2);
    for (int i = 0; i < 20; ++i) state.cards.createTempCardInDrawPile(0, CardInstance{CardId::DEFEND_RED});
    Action{search::ActionType::END_TURN}.execute(state);
    require(state.inputState == InputState::CARD_SELECT, "turn start must pause for the first discard");
    pickAndCompare(state);
    require(state.inputState == InputState::CARD_SELECT, "each Tools of the Trade stack must get a choice");
    pickAndCompare(state);
    require(state.inputState == InputState::PLAYER_NORMAL && state.resolution.empty(), "turn start did not finish");
}

void openingChoices() {
    GameContext game{CharacterClass::IRONCLAD, 31, 20};
    game.relics.add({RelicId::TOOLBOX});
    game.relics.add({RelicId::GAMBLING_CHIP});
    BattleContext state;
    state.init(game, MonsterEncounter::JAW_WORM);
    require(state.cardSelectInfo.cardSelectTask == CardSelectTask::DISCOVERY, "Toolbox must open first");
    pickAndCompare(state);
    require(state.inputState == InputState::CARD_SELECT && state.cardSelectInfo.cardSelectTask == CardSelectTask::GAMBLE,
            "Gambling Chip must not overwrite the Toolbox choice");
    pickAndCompare(state);
    require(state.inputState == InputState::PLAYER_NORMAL && state.resolution.empty(), "opening choices did not finish");
}

// The walk names everything play depends on: a copy filled from it plays on exactly like the original.
void copiesPlayAlike() {
    const MonsterEncounter encounters[] = {
        MonsterEncounter::JAW_WORM, MonsterEncounter::GREMLIN_NOB, MonsterEncounter::THREE_SENTRIES,
        MonsterEncounter::SLIME_BOSS, MonsterEncounter::GREMLIN_LEADER, MonsterEncounter::CHAMP,
        MonsterEncounter::AUTOMATON, MonsterEncounter::COLLECTOR, MonsterEncounter::REPTOMANCER,
        MonsterEncounter::AWAKENED_ONE, MonsterEncounter::TIME_EATER, MonsterEncounter::DONU_AND_DECA};
    std::mt19937_64 random{11};
    int copies = 0, choices = 0;
    for (std::uint64_t seed = 1; seed <= 48; ++seed) {
        GameContext game{CharacterClass::IRONCLAD, seed, 20};
        game.floorNum = 30;
        for (int i = 0; i < 15; ++i) {  // a varied deck, so play reaches many kinds of state
            Card card{CombatCardPool::getCardAt(CharacterClass::IRONCLAD, static_cast<int>(random() % CombatCardPool::getPoolSize(CharacterClass::IRONCLAD)))};
            card.upgraded = random() % 2;
            game.deck.obtainRaw(card);
        }
        BattleContext state;
        state.init(game, encounters[seed % std::size(encounters)]);
        while (state.outcome == Outcome::UNDECIDED && state.turn < 20) {
            choices += state.inputState == InputState::CARD_SELECT;
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

            randomAction(state, random).execute(state);
        }
    }
    require(choices > 0, "the fights must exercise suspended choices");
    require(copies > 200, "the fights must reach many states");
}
}  // namespace

int main() {
    try {
        cardChoices();
        nestedAutoplay();
        exhaustBatch();
        turnChoices();
        openingChoices();
        copiesPlayAlike();
        std::cout << "state fields tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "state fields test failed: " << e.what() << '\n';
        return 1;
    }
}
