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
template <class State> std::vector<std::uint64_t> leaves(const State &state) {
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
template <class State = BattleContext> State filled(const std::vector<std::uint64_t> &words) {
    State state;
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

template <class Effect> void payloadRoundTrip() {
    Effect effect{};
    forEachField(effect, [](auto& leaf) {
        using T = std::decay_t<decltype(leaf)>;
        leaf = static_cast<T>(1);
    });
    const sts::Action original{std::move(effect), false};
    const auto words = leaves(original);
    require(leaves(filled<sts::Action>(words)) == words, "action payload did not round trip");
}

template <std::size_t... I> void allPayloadsRoundTrip(std::index_sequence<I...>) {
    (payloadRoundTrip<std::variant_alternative_t<I, ActionPayload>>(), ...);
}

BattleContext fixture() {
    GameContext game{CharacterClass::IRONCLAD, 17, 20};
    BattleContext state;
    state.init(game, MonsterEncounter::JAW_WORM);
    state.actionQueue.clear();
    state.cards = CardManager{};
    return state;
}

void queueOrderAndWrap() {
    ActionQueue<3> queue;
    auto state = fixture();
    const auto energy = state.player.energy;
    int gained = 0;
    for (int amount = 1; amount < 9; ++amount) {
        queue.pushBack(Actions::GainEnergy(amount));
        queue.popFront().execute(state);
        gained += amount;
    }
    queue.pushBack(Actions::GainEnergy(8));
    queue.pushFront(Actions::GainEnergy(9));
    queue.pushBack(Actions::GainEnergy(10));
    auto copy = filled<ActionQueue<3>>(leaves(queue));
    for (int amount : {9, 8, 10}) {
        const auto expected = leaves(Actions::GainEnergy(amount));
        require(leaves(queue.popFront()) == expected && leaves(copy.popFront()) == expected,
                "restored queue changed order");
    }
    require(queue.isEmpty() && copy.isEmpty(), "queue failed to drain");
    queue.pushBack(Actions::GainEnergy(100));
    queue.clear();
    require(queue.isEmpty(), "clear must discard pending actions");
    queue.pushBack(Actions::GainEnergy(2));
    queue.popFront().execute(state);
    require(state.player.energy == energy + gained + 2, "cleared queue could not be reused");
}

void pendingRetainedCards() {
    auto state = fixture();
    state.cards.limbo[0] = CardInstance{CardId::BASH, true};
    state.cards.limbo[0].uniqueId = 42;
    state.cards.limbo[0].retain = true;
    state.addToBot(Actions::RestoreRetainedCards(1));
    state.addToBot(Actions::SetState(InputState::PLAYER_NORMAL));
    state.inputState = InputState::EXECUTING_ACTIONS;
    auto copy = filled(leaves(state));
    state.executeActions();
    copy.executeActions();
    require(state.cards.cardsInHand == 1 && state.cards.hand[0].uniqueId == 42,
            "retained card was not restored");
    require(leaves(state) == leaves(copy), "restored retained-card action lost its card");
}

void pendingSelection() {
    auto state = fixture();
    state.cards.createTempCardInDiscard(CardInstance{CardId::BASH});
    state.cards.createTempCardInDiscard(CardInstance{CardId::DEFEND_RED});
    state.addToBot(Actions::HeadbuttAction());
    state.addToBot(Actions::GainEnergy(3));
    state.addToBot(Actions::SetState(InputState::PLAYER_NORMAL));
    state.inputState = InputState::EXECUTING_ACTIONS;
    state.executeActions();
    require(state.inputState == InputState::CARD_SELECT && state.actionQueue.size > 0,
            "selection must pause with queued work");
    auto copy = filled(leaves(state));
    const auto action = search::Action::getAllActionsInState(state).front();
    action.execute(state);
    action.execute(copy);
    require(leaves(state) == leaves(copy), "restored selection resumed differently");
}

void victoryFlags() {
    auto state = fixture();
    state.player.curHp = state.player.maxHp - 5;
    const int energy = state.player.energy;
    state.addToBot(Actions::GainEnergy(7));
    state.addToBot(Actions::HealPlayer(3));
    state.outcome = Outcome::PLAYER_VICTORY;
    auto copy = filled(leaves(state));
    for (auto* battle : {&state, &copy}) {
        battle->clearPostCombatActions();
        battle->inputState = InputState::EXECUTING_ACTIONS;
        battle->executeActions();
        require(battle->player.energy == energy && battle->player.curHp == battle->player.maxHp - 2,
                "victory changed clear/survive behavior");
    }
    require(leaves(state) == leaves(copy), "restored queue changed victory behavior");
}

void formerlyInlineEffects() {
    for (int hp : {29, 31}) {
        auto state = fixture();
        state.monsters.arr[0].curHp = hp;
        state.addToBot(Actions::Judgment(0, 30));
        state.addToBot(Actions::SetState(InputState::PLAYER_NORMAL));
        state.inputState = InputState::EXECUTING_ACTIONS;
        auto copy = filled(leaves(state));
        state.executeActions();
        copy.executeActions();
        require(state.monsters.arr[0].isDeadOrEscaped() == (hp <= 30), "Judgment changed its threshold");
        require(leaves(state) == leaves(copy), "restored Judgment changed behavior");
    }
    auto state = fixture();
    state.monsters.arr[0].addDebuff<MS::POISON>(3, false);
    state.addToBot(Actions::Catalyst(0, 3));
    state.addToBot(Actions::SetState(InputState::PLAYER_NORMAL));
    state.inputState = InputState::EXECUTING_ACTIONS;
    auto copy = filled(leaves(state));
    state.executeActions();
    copy.executeActions();
    require(state.monsters.arr[0].getStatus<MS::POISON>() == 9, "Catalyst changed its multiplier");
    require(leaves(state) == leaves(copy), "restored Catalyst changed behavior");
}

// The walk names everything play depends on: a copy filled from it plays on exactly like the original.
void copiesPlayAlike() {
    const MonsterEncounter encounters[] = {
        MonsterEncounter::JAW_WORM, MonsterEncounter::GREMLIN_NOB, MonsterEncounter::THREE_SENTRIES,
        MonsterEncounter::SLIME_BOSS, MonsterEncounter::GREMLIN_LEADER, MonsterEncounter::CHAMP,
        MonsterEncounter::AUTOMATON, MonsterEncounter::COLLECTOR, MonsterEncounter::REPTOMANCER,
        MonsterEncounter::AWAKENED_ONE, MonsterEncounter::TIME_EATER, MonsterEncounter::DONU_AND_DECA};
    std::mt19937_64 random{11};
    int copies = 0, queued = 0;
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
            queued += state.actionQueue.size > 0;
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
    require(queued > 0, "the fights must exercise queued states");
    require(copies > 200, "the fights must reach many states");
}
}  // namespace

int main() {
    try {
        allPayloadsRoundTrip(std::make_index_sequence<std::variant_size_v<ActionPayload>>{});
        queueOrderAndWrap();
        pendingRetainedCards();
        pendingSelection();
        victoryFlags();
        formerlyInlineEffects();
        copiesPlayAlike();
        std::cout << "state fields tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "state fields test failed: " << e.what() << '\n';
        return 1;
    }
}
