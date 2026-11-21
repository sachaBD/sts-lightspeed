//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_ML_BATTLECONTEXT_H
#define STS_ML_BATTLECONTEXT_H

#include <vector>
#include <array>

#include "sts_common.h"

#include "data_structure/fixed_list.h"
#include "constants/Potions.h"
#include "constants/MonsterEncounters.h"
#include "combat/InputState.h"
#include "combat/Player.h"
#include "combat/Monster.h"
#include "combat/MonsterGroup.h"
#include "combat/Resolution.h"
#include "combat/CardPlay.h"
#include "combat/Actions.h"
#include "combat/CardManager.h"
#include "combat/CardSelectInfo.h"


namespace sts {



    enum class Outcome {
        UNDECIDED=0,
        PLAYER_VICTORY,
        PLAYER_LOSS,
    };

    // The simulator deliberately records an effect that it cannot model,
    // rather than continuing with a silently incorrect state transition.
    // RLEnvironment exposes this as an explicit episode truncation.
    enum class UnsupportedEffectKind {
        NONE=0,
        CARD,
        POTION,
        SYSTEM,
    };

    static constexpr const char * battleOutcomeStrings[] {
            "UNDECIDED",
            "PLAYER_VICTORY",
            "PLAYER_LOSS",
    };

    struct GameContext;

    extern thread_local BattleContext *g_debug_bc;

    struct BattleContext {

        // begin for debugging purposes
        inline static int sum = 0; // for preventing optimization in benchmarks
        bool haveUsedDiscoveryAction = false; // for tracking undefined behavior resulting from using the action
        bool undefinedBehaviorEvoked = false; // some cards cause inconsistent outcomes in games
        std::uint64_t seed = 0;
        int floorNum = 0;
        MonsterEncounter encounter = MonsterEncounter::INVALID;
        int loopCount = 0;
        int energyWasted = 0;
        int cardsDrawn = 0;
        // Diagnostic only, excluded from combat observations. Deltas scope reporting to an execution,
        // without leaking earlier accesses through copied states or calls outside Action::execute.
        std::uint64_t hiddenInformationAccesses = 0;
        // end for debugging purposes

        Random aiRng;
        Random cardRandomRng;
        Random miscRng;
        Random monsterHpRng;
        Random potionRng;
        Random shuffleRng;

        int ascension = 0;
        Outcome outcome = Outcome::UNDECIDED;
        UnsupportedEffectKind unsupportedEffectKind = UnsupportedEffectKind::NONE;
        int unsupportedEffectId = 0;
        InputState inputState = InputState::EXECUTING_ACTIONS;
        CardSelectInfo cardSelectInfo;

        int monsterTurnIdx = 6;

        bool isBattleOver = false;
        bool escapedCombat = false;
        bool endTurnRequested = false;
        bool endTurnAfterCurrentCard = false;
        bool turnHasEnded = false;
        bool skipMonsterTurn = false;

        std::vector<ResolutionFrame> resolution;

        // Pending Nightmare copies, resolved at the start of the next player
        // turn after the agent has selected the source hand card.
        fixed_list<CardInstance, 10> nightmareCards;
        fixed_list<int, 10> nightmareCopyCounts;

        int potionCount = 0;
        int potionCapacity = 3;
        std::array<Potion, 5> potions;

        int turn = 0;
        Player player;
        MonsterGroup monsters;
        CardManager cards;

        CardPlay currentCard;

        std::bitset<32> miscBits; // 0 stolen gold check,

        // Every field, in a fixed order, for code that stores states: see combat/StateFields.h.
        template <class Self, class Visit> static void visitFields(Self &s, Visit &&visit) {
            visit(s.haveUsedDiscoveryAction); visit(s.undefinedBehaviorEvoked); visit(s.seed); visit(s.floorNum);
            visit(s.encounter); visit(s.loopCount); visit(s.energyWasted); visit(s.cardsDrawn);
            visit(s.hiddenInformationAccesses);
            visit(s.aiRng); visit(s.cardRandomRng); visit(s.miscRng); visit(s.monsterHpRng); visit(s.potionRng);
            visit(s.shuffleRng);
            visit(s.ascension); visit(s.outcome); visit(s.unsupportedEffectKind); visit(s.unsupportedEffectId);
            visit(s.inputState); visit(s.cardSelectInfo); visit(s.monsterTurnIdx); visit(s.isBattleOver);
            visit(s.escapedCombat); visit(s.endTurnRequested); visit(s.endTurnAfterCurrentCard); visit(s.turnHasEnded);
            visit(s.skipMonsterTurn); visit(s.resolution); visit(s.nightmareCards); visit(s.nightmareCopyCounts);
            visit(s.potionCount); visit(s.potionCapacity); visit(s.potions); visit(s.turn); visit(s.player);
            visit(s.monsters); visit(s.cards); visit(s.currentCard); visit(s.miscBits);
        }

        BattleContext() = default;
        BattleContext(const BattleContext &rhs) = default;

// ****************************************

        void init(const GameContext &gc);
        void init(const GameContext &gc, MonsterEncounter encounterToInit);

        void initRelics(const GameContext &gc);

        void exitBattle(GameContext &g) const;
        void updateRelicsOnExit(GameContext &g) const;
        void updateCardsOnExit(Deck &d) const; // for cards like ritual dagger, and eventually lesson learned results

// ****************************************

        void setRequiresStolenGoldCheck(bool value);
        [[nodiscard]] bool requiresStolenGoldCheck() const;
        [[nodiscard]] int getMonsterTurnNumber() const;  // returns 1 for first turn, 2 for second, etc.

// ****************************************

        void executeActions();

        void beginCard(CardPlay);
        bool prepareCard();
        void drawAfterChoice(int);
        void finishStartTurn();
        void useNoTriggerCard();

        void useAttackCard();
        void useSkillCard();
        void usePowerCard();

        void onUseAttackCard();
        void onUseSkillCard();
        void onUsePowerCard();

        void onUseStatusOrCurseCard();
        void onAfterUseCard();

        void setState(InputState state);


        void checkCombat();

        [[nodiscard]] bool isCardPlayAllowed() const;

        // **********************

        void endTurn(); // called when player clicks the end turn button
        void callEndOfTurnActions();
        void onTurnEnding(); // AbstractRoom endTurn(), called when the turn actually ends.

        void callEndTurnEarlySequence(); // time eater

        void applyEndOfRoundPowers(); // game : MonsterGroup applyEndOfTurnPowers
        void afterMonsterTurns();

        void obtainPotion(Potion p);
        void discardPotion(int idx);
        void drinkPotion(int idx, int target=0);

        void noteDrawPileOrderAccess();
        void noteEnemyIntentAccess();
        void drawCards(int count);
        void discardAtEndOfTurn();

        void playTopCardInDrawPile(int monsterTargetIdx, bool exhausts);

        void moveToHandHelper(CardInstance c);
        void exhaustSpecificCardInHand(int idx, std::int16_t uniqueId);
        void exhaustTopCardInHand();

        // this is
        void triggerOnEndOfTurnForPlayingCards();
        void triggerOnOtherCardPlayed(const CardInstance &usedCard);

        template <MonsterStatus s>
        void debuffEnemy(int idx, int amount, bool isSourceMonster=true);

        [[nodiscard]] int calculateCardDamage(const CardInstance &card, int targetIdx, int baseDamage) const;
        [[nodiscard]] int calculateCardBlock(int baseBlock) const;

        void queuePurgeCard(const CardInstance &c, int target);
        void noOpRollMove(); // called by monsters to manipulate the aiRng counter when their rollMove function doesn't do anything

        void onManualDiscard(const CardInstance &c);
        void onShuffle();
        void triggerAndMoveToExhaustPile(CardInstance c);
        void mummifiedHandOnUsePower();

        // card select screens
        void openDiscoveryScreen(std::array<CardId, 3> discoveryCards, int copyCount);
        void openSimpleCardSelectScreen(CardSelectTask task, int count);

        // single card select helpers
        void chooseArmamentsCard(int handIdx);
        void chooseCodexCard(CardId id);
        void chooseDiscardToHandCard(int discardIdx, bool forZeroCost);
        void chooseDiscoveryCard(CardId id);
        void chooseDualWieldCard(int handIdx);
        void chooseExhaustOneCard(int handIdx);
        void chooseExhumeCard(int exhaustIdx);
        void chooseForethoughtCard(int handIdx);
        void chooseDiscardCard(int handIdx);
        void chooseHeadbuttCard(int discardIdx);
        void chooseRecycleCard(int handIdx);
        void chooseSeekCard(int drawIdx);
        void chooseOmniscienceCard(int drawIdx);
        void chooseNightmareCard(int handIdx);
        void chooseScryCards(const fixed_list<int, 10> &idxs);
        void chooseWish(int choice);
        void chooseSetupCard(int handIdx);
        void chooseWarcryCard(int handIdx);

        // multi card helpers
        void chooseDrawToHandCards(const int *idxs, int cardCount);
        void chooseExhaustCards(const fixed_list<int,10> &idxs);
        void chooseGambleCards(const fixed_list<int,10> &idxs);

    };

    std::ostream& operator<<(std::ostream &os, const BattleContext &bc);

    template <PlayerStatus s>
    void Actions::BuffPlayer(BattleContext& bc, int amount) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        if (s == PlayerStatus::CORRUPTION && !bc.player.hasStatus<PS::CORRUPTION>()) {
            bc.cards.onBuffCorruption();
        }
        bc.player.buff<s>(amount);
    }

    template <PlayerStatus s>
    void Actions::DebuffPlayer(BattleContext& bc, int amount, bool isSourceMonster) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        bc.player.debuff<s>(amount, isSourceMonster);
    }

    template<PlayerStatus s>
    void Actions::DecrementStatus(BattleContext& bc, int amount) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        bc.player.decrementStatus<s>(amount);
    }

    template <PlayerStatus s>
    void Actions::RemoveStatus(BattleContext& bc) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        bc.player.setHasStatus<s>(false);
    }

    template<MonsterStatus s>
    void Actions::BuffEnemy(BattleContext& bc, int idx, int amount) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        // todo check if alive?
        bc.monsters.arr[idx].buff<s>(amount);
    }

    template <MonsterStatus s>
    void Actions::DebuffEnemy(BattleContext& bc, int idx, int amount, bool isSourceMonster) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        bc.debuffEnemy<s>(idx, amount, isSourceMonster);
    }

    template <MonsterStatus s>
    void Actions::DebuffAllEnemy(BattleContext& bc, int amount, bool isSourceMonster) {
        if (bc.outcome != Outcome::UNDECIDED) return;
        for (int i = bc.monsters.monsterCount-1; i >= 0; --i) {
            if (bc.monsters.arr[i].isTargetable()) {
                Actions::DebuffEnemy<s>(bc, i, amount, isSourceMonster);
            }
        }
    }

    template <MonsterStatus s>
    void BattleContext::debuffEnemy(int idx, int amount, bool isSourceMonster) {
        // todo poison and snake skull
        Monster &e = monsters.arr[idx];

        if (e.hasStatus<MS::ARTIFACT>()) {
            e.decrementStatus<MS::ARTIFACT>();
            return;
        }

        e.addDebuff<s>(amount, isSourceMonster);
        if (player.hasStatus<PS::SADISTIC>()) {
            Actions::DamageEnemy(*this, idx, player.getStatus<PS::SADISTIC>());
        }

        if (s == MS::VULNERABLE && player.hasRelic<RelicId::CHAMPION_BELT>()) {
            Actions::DebuffEnemy<MS::WEAK>(*this, idx, 1, false);
        }
    }


}

#endif //STS_ML_BATTLECONTEXT_H
