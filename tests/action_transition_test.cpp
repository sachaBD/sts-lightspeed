#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "sim/search/Action.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace sts;
using search::Action;
using search::ActionType;
using search::TransitionEffects;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error{message};
}

BattleContext fixture(CardId card, std::uint64_t seed = 7) {
    GameContext game{CharacterClass::IRONCLAD, seed, 1};
    game.floorNum = 1;
    BattleContext state;
    state.init(game, MonsterEncounter::JAW_WORM);
    state.cards = CardManager{};
    state.cards.createTempCardInHand(CardInstance{card});
    state.cards.createTempCardInDrawPile(0, CardInstance{CardId::BASH});
    state.cards.createTempCardInDrawPile(1, CardInstance{CardId::DEFEND_RED});
    return state;
}

TransitionEffects play(BattleContext &state) {
    const Action action{ActionType::CARD, 0, 0};
    require(action.isValidAction(state), "fixture card must be legal");
    return action.execute(state);
}

void deterministicAcrossWorlds() {
    for (std::uint64_t seed = 1; seed <= 16; ++seed) {
        auto first = fixture(CardId::STRIKE_RED, seed);
        auto second = first;
        std::reverse(second.cards.drawPile.begin(), second.cards.drawPile.end());
        second.aiRng = Random(seed + 100);
        second.cardRandomRng = Random(seed + 200);
        require(!play(first).isChance() && !play(second).isChance(), "plain Strike must be deterministic");
        require(search::PublicBeliefCombatSearch::observationKey(first)
                    == search::PublicBeliefCombatSearch::observationKey(second),
                "deterministic transitions must agree across compatible worlds");
    }
}

void hiddenDrawWithoutRng() {
    auto first = fixture(CardId::SHRUG_IT_OFF);
    auto second = first;
    std::reverse(second.cards.drawPile.begin(), second.cards.drawPile.end());
    for (auto *state : {&first, &second}) {
        const auto effects = play(*state);
        require(effects.accessedHiddenInformation && !effects.consumedRng && effects.isChance(),
                "drawing a pre-shuffled hidden card must be chance without consuming RNG");
    }
    require(search::PublicBeliefCombatSearch::observationKey(first)
                != search::PublicBeliefCombatSearch::observationKey(second),
            "different hidden draws must produce different observations");
}

void visibleAndBlockedDraws() {
    auto visible = fixture(CardId::SHRUG_IT_OFF);
    visible.player.setHasRelic<RelicId::FROZEN_EYE>(true);
    require(!play(visible).isChance(), "Frozen Eye draw without RNG must be deterministic");

    auto blocked = fixture(CardId::SHRUG_IT_OFF);
    blocked.player.debuff<PS::NO_DRAW>(1);
    require(!play(blocked).isChance(), "blocked draw must not report a hidden access");

    auto confused = fixture(CardId::SHRUG_IT_OFF);
    confused.player.setHasRelic<RelicId::FROZEN_EYE>(true);
    confused.player.debuff<PS::CONFUSED>(1);
    const auto effects = play(confused);
    require(effects.consumedRng && !effects.accessedHiddenInformation && effects.isChance(),
            "visible draw with randomized cost must still be chance");

    auto full = fixture(CardId::STRIKE_RED);
    while (full.cards.cardsInHand < CardManager::MAX_HAND_SIZE)
        full.cards.createTempCardInHand(CardInstance{CardId::DEFEND_RED});
    full.addToBot(Actions::DrawCards(1));
    require(!play(full).isChance(), "hand-limit-blocked queued draw must be deterministic");
}

void allRngStreamsAndLowLevelCalls() {
    const std::array<Random BattleContext::*, 6> streams{
        &BattleContext::aiRng, &BattleContext::cardRandomRng, &BattleContext::miscRng,
        &BattleContext::monsterHpRng, &BattleContext::potionRng, &BattleContext::shuffleRng};
    for (auto stream : streams) {
        for (bool lowLevel : {false, true}) {
            auto state = fixture(CardId::STRIKE_RED);
            state.addToBot({[=](BattleContext &bc) {
                if (lowLevel) (bc.*stream).nextLong();
                else (bc.*stream).random(0);  // Only one possible result still consumes RNG.
            }});
            const auto effects = play(state);
            require(effects.consumedRng && !effects.accessedHiddenInformation && effects.isChance(),
                    "all RNG streams and low-level calls must be detected through queued effects");
        }
    }
}

void shuffleAndRandomEffects() {
    auto shuffled = fixture(CardId::SHRUG_IT_OFF);
    shuffled.player.setHasRelic<RelicId::FROZEN_EYE>(true);
    shuffled.cards.drawPile.clear();
    shuffled.cards.createTempCardInDiscard(CardInstance{CardId::BASH});
    shuffled.cards.createTempCardInDiscard(CardInstance{CardId::DEFEND_RED});
    require(play(shuffled).consumedRng, "reshuffling must be chance even with Frozen Eye");

    auto targeted = fixture(CardId::SWORD_BOOMERANG);
    require(play(targeted).consumedRng, "random targets must be chance mid-turn");

    auto generated = fixture(CardId::DISCOVERY);
    require(play(generated).consumedRng, "random generation must be chance before selection");
    require(generated.inputState == InputState::CARD_SELECT, "Discovery must pause at the selection");
    const auto actions = Action::getAllActionsInState(generated);
    require(!actions.empty(), "Discovery must offer a selection");
    require(!actions.front().execute(generated).isChance(), "choosing a public offer must not inherit RNG use");
}

void scryAndTopCardEffects() {
    for (bool frozenEye : {false, true}) {
        auto scry = fixture(CardId::STRIKE_RED);
        scry.player.setHasRelic<RelicId::FROZEN_EYE>(frozenEye);
        scry.addToBot(Actions::ScryAction(1));
        const auto effects = play(scry);
        require(!effects.consumedRng && effects.accessedHiddenInformation == !frozenEye,
                "Scry must report revealing hidden order, unless Frozen Eye exposes it");
        require(scry.inputState == InputState::CARD_SELECT, "Scry must pause at its public menu");
        require(!Action{ActionType::MULTI_CARD_SELECT, 1}.execute(scry).isChance(),
                "discarding a publicly offered Scry card must be deterministic");

        auto autoplay = fixture(CardId::STRIKE_RED);
        autoplay.addToBot(Actions::PlayTopCard(0, true));
        autoplay.player.setHasRelic<RelicId::FROZEN_EYE>(frozenEye);
        const auto top = play(autoplay);
        require(!top.consumedRng && top.accessedHiddenInformation == !frozenEye,
                "top-card autoplay must report hidden-order dependence without RNG");
    }

    auto peek = fixture(CardId::ESCAPE_PLAN);
    peek.player.debuff<PS::NO_DRAW>(1);
    const auto effects = play(peek);
    require(effects.accessedHiddenInformation && !effects.consumedRng,
            "top-card inspection must count even when the subsequent draw is blocked");
}

void hiddenIntents() {
    for (auto card : {CardId::SPOT_WEAKNESS, CardId::GO_FOR_THE_EYES, CardId::FEAR_NO_EVIL}) {
        for (bool dome : {false, true}) {
            auto state = fixture(card);
            state.player.setHasRelic<RelicId::RUNIC_DOME>(dome);
            const auto effects = play(state);
            require(!effects.consumedRng && effects.accessedHiddenInformation == dome,
                    "intent-dependent effects must count hidden intent under Runic Dome");
        }
    }
}

void copiesAndEarlierAccessesDoNotLeak() {
    auto diagnostic = fixture(CardId::STRIKE_RED);
    const auto key = search::PublicBeliefCombatSearch::observationKey(diagnostic);
    diagnostic.noteDrawPileOrderAccess();
    require(key == search::PublicBeliefCombatSearch::observationKey(diagnostic),
            "reporting diagnostics must not change combat observations");

    auto state = fixture(CardId::STRIKE_RED);
    state.drawCards(1);  // Outside the measured transition.
    auto copy = state;
    require(!play(state).isChance() && !play(copy).isChance(),
            "prior hidden accesses must not leak through original or copied states");
}

void emptyPileEffects() {
    auto scry = fixture(CardId::STRIKE_RED);
    scry.cards.drawPile.clear();
    scry.addToBot(Actions::ScryAction(1));
    require(!play(scry).isChance(), "Scry over an empty pile reveals nothing");

    auto autoplay = fixture(CardId::STRIKE_RED);
    autoplay.cards.drawPile.clear();
    autoplay.addToBot(Actions::PlayTopCard(0, true));
    require(!play(autoplay).isChance(), "autoplay with no draw or discard cards accesses no hidden card");
}

void endTurnAndTerminal() {
    auto state = fixture(CardId::STRIKE_RED);
    require(Action{ActionType::END_TURN}.execute(state).isChance(),
            "End Turn must include next intents and next-turn draws");
    require(state.turn > 0 && state.inputState == InputState::PLAYER_NORMAL,
            "End Turn reporting must cover the full transition to the next input");

    auto terminal = fixture(CardId::STRIKE_RED);
    terminal.monsters.arr[0].curHp = 1;
    require(!play(terminal).isChance() && terminal.outcome == Outcome::PLAYER_VICTORY,
            "deterministic terminal actions must also return a report");
}
}  // namespace

int main() {
    try {
        deterministicAcrossWorlds();
        hiddenDrawWithoutRng();
        visibleAndBlockedDraws();
        allRngStreamsAndLowLevelCalls();
        shuffleAndRandomEffects();
        scryAndTopCardEffects();
        hiddenIntents();
        copiesAndEarlierAccessesDoNotLeak();
        emptyPileEffects();
        endTurnAndTerminal();
        std::cout << "Action transition invariants passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
