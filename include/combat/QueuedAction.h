#pragma once
#include "combat/CardQueue.h"
#include "combat/CardSelectInfo.h"
#include "combat/InputState.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include <array>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace sts {
struct BattleContext;
using DamageMatrix = std::array<std::uint16_t, 5>;

namespace queued {
struct NoOp {
    void execute(BattleContext&) const {}
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SetState {
    InputState state{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.state);
    }
};

struct AttackEnemy {
    int idx{};
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage);
    }
};

struct AttackAllEnemy {
    int baseDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.baseDamage);
    }
};

struct AttackAllEnemyMatrix {
    DamageMatrix damageMatrix{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.damageMatrix);
    }
};

struct DamageEnemy {
    int idx{};
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage);
    }
};

struct DamageAllEnemy {
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.damage);
    }
};

struct AttackPlayer {
    int idx{};
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage);
    }
};

struct DamagePlayer {
    int damage{};
    bool selfDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.damage); visit(s.selfDamage);
    }
};

struct VampireAttack {
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.damage);
    }
};

struct PlayerLoseHp {
    int hp{};
    bool selfDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.hp); visit(s.selfDamage);
    }
};

struct HealPlayer {
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount);
    }
};

struct MonsterGainBlock {
    int idx{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.amount);
    }
};

struct RollMove {
    int monsterIdx{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.monsterIdx);
    }
};

struct ReactiveRollMove {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct NoOpRollMove {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ChangeStance {
    Stance stance{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.stance);
    }
};

struct GainEnergy {
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount);
    }
};

struct GainBlock {
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount);
    }
};

struct DrawCards {
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount);
    }
};

struct EmptyDeckShuffle {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ShuffleDrawPile {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ShuffleTempCardIntoDrawPile {
    CardId id{};
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.id); visit(s.count);
    }
};

struct PlayTopCard {
    int monsterTargetIdx{};
    bool exhausts{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.monsterTargetIdx); visit(s.exhausts);
    }
};

struct MakeTempCardInHand {
    CardInstance card{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.card); visit(s.amount);
    }
};

struct MakeTempCardInDrawPile {
    CardInstance c{};
    int amount{};
    bool shuffleInto{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.c); visit(s.amount); visit(s.shuffleInto);
    }
};

struct MakeTempCardInDiscard {
    CardInstance c{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.c); visit(s.amount);
    }
};

struct MakeTempCardsInHand {
    std::vector<CardInstance> cards{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.cards);
    }
};

struct DiscardNoTriggerCard {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ClearCardQueue {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DiscardAtEndOfTurn {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DiscardAtEndOfTurnHelper {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct RestoreRetainedCards {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct UnnamedEndOfTurnAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct MonsterStartTurnAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct TriggerEndOfTurnOrbsAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ChannelOrb {
    Orb orb{};
    int orbData{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.orb); visit(s.orbData);
    }
};

struct EvokeFrontOrb {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ExhaustTopCardInHand {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ExhaustSpecificCardInHand {
    int idx{};
    std::int16_t uniqueId{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.uniqueId);
    }
};

struct DamageRandomEnemy {
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.damage);
    }
};

struct ExhaustRandomCardInHand {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct MadnessAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct RandomizeHandCost {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct GainBlockRandomEnemy {
    int sourceMonster{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.sourceMonster); visit(s.amount);
    }
};

struct SummonGremlins {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SpawnTorchHeads {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SpireShieldDebuff {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct OnAfterCardUsed {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct PutRandomCardsInDrawPile {
    CardType type{};
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.type); visit(s.count);
    }
};

struct DiscoveryAction {
    CardType type{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.type); visit(s.amount);
    }
};

struct InfernalBladeAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct JackOfAllTradesAction {
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.upgraded);
    }
};

struct TransmutationAction {
    bool upgraded{};
    int energy{};
    bool useEnergy{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.upgraded); visit(s.energy); visit(s.useEnergy);
    }
};

struct ViolenceAction {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct BetterDiscardPileToHandAction {
    int amount{};
    CardSelectTask task{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount); visit(s.task);
    }
};

struct ArmamentsAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DualWieldAction {
    int copyCount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.copyCount);
    }
};

struct ExhumeAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ForethoughtAction {
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.upgraded);
    }
};

struct HeadbuttAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ChooseExhaustOne {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DrawToHandAction {
    CardSelectTask task{};
    CardType cardType{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.task); visit(s.cardType);
    }
};

struct WarcryAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct TimeEaterPlayCardQueueItem {
    CardQueueItem x{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.x);
    }
};

struct UpgradeAllCardsInHand {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct EssenceOfDarkness {
    int darkOrbsPerSlot{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.darkOrbsPerSlot);
    }
};

struct MakeRandomCardInHand {
    CardType type{};
    int count{};
    bool zeroCost{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.type); visit(s.count); visit(s.zeroCost);
    }
};

struct MakeRandomCommonCardInHand {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct DiscardOne {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct RecycleAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ScrapeAction {
    int drawCount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.drawCount);
    }
};

struct RebootAction {
    int drawCount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.drawCount);
    }
};

struct SeekAction {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct AllForOneAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct IncreaseOrbSlots {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct SuicideAction {
    int monsterIdx{};
    bool triggerRelics{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.monsterIdx); visit(s.triggerRelics);
    }
};

struct PoisonLoseHpAction {
    int monsterIdx{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.monsterIdx);
    }
};

struct RemovePlayerDebuffs {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct UpgradeRandomCardAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct CodexAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ExhaustMany {
    int limit{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.limit);
    }
};

struct GambleAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ToolboxAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DualityAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ApotheosisAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct DropkickAction {
    int targetIdx{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.targetIdx);
    }
};

struct EnlightenmentAction {
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.upgraded);
    }
};

struct EntrenchAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct FeedAction {
    int idx{};
    int damage{};
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage); visit(s.upgraded);
    }
};

struct FiendFireAction {
    int targetIdx{};
    int calculatedDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.targetIdx); visit(s.calculatedDamage);
    }
};

struct HandOfGreedAction {
    int idx{};
    int damage{};
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage); visit(s.upgraded);
    }
};

struct LimitBreakAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct ReaperAction {
    int baseDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.baseDamage);
    }
};

struct RitualDaggerAction {
    int idx{};
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.idx); visit(s.damage);
    }
};

struct SecondWindAction {
    int blockPerCard{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.blockPerCard);
    }
};

struct SeverSoulExhaustAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SwordBoomerangAction {
    int baseDamage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.baseDamage);
    }
};

struct SpotWeaknessAction {
    int target{};
    int strength{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.target); visit(s.strength);
    }
};

struct ScryAction {
    int count{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.count);
    }
};

struct BouncingFlaskAction {
    int poison{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.poison);
    }
};

struct CalculatedGambleAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct EscapePlanAction {
    int block{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.block);
    }
};

struct RandomDiscardAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SetupAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct SunderAction {
    int target{};
    int damage{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.target); visit(s.damage);
    }
};

struct ModifyCurrentCardCost {
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.amount);
    }
};

struct StormOfSteelAction {
    bool upgraded{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.upgraded);
    }
};

struct BulletTimeAction {
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self&, Visit&&) {}
};

struct WhirlwindAction {
    int baseDamage{};
    int energy{};
    bool useEnergy{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.baseDamage); visit(s.energy); visit(s.useEnergy);
    }
};

struct AttackAllMonsterRecursive {
    DamageMatrix matrix{};
    int timesRemaining{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.matrix); visit(s.timesRemaining);
    }
};

struct Judgment {
    int target{};
    int threshold{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.target); visit(s.threshold);
    }
};

struct Catalyst {
    int target{};
    int multiplier{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.target); visit(s.multiplier);
    }
};

struct BuffPlayer {
    PlayerStatus status{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.amount);
    }
};

struct DebuffPlayer {
    PlayerStatus status{};
    int amount{};
    bool isSourceMonster{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.amount); visit(s.isSourceMonster);
    }
};

struct DecrementStatus {
    PlayerStatus status{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.amount);
    }
};

struct RemoveStatus {
    PlayerStatus status{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status);
    }
};

struct BuffEnemy {
    MonsterStatus status{};
    int idx{};
    int amount{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.idx); visit(s.amount);
    }
};

struct DebuffEnemy {
    MonsterStatus status{};
    int idx{};
    int amount{};
    bool isSourceMonster{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.idx); visit(s.amount); visit(s.isSourceMonster);
    }
};

struct DebuffAllEnemy {
    MonsterStatus status{};
    int amount{};
    bool isSourceMonster{};
    void execute(BattleContext&) const;
    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.status); visit(s.amount); visit(s.isSourceMonster);
    }
};

} // namespace queued

using ActionPayload = std::variant<
    queued::NoOp,
    queued::SetState,
    queued::AttackEnemy,
    queued::AttackAllEnemy,
    queued::AttackAllEnemyMatrix,
    queued::DamageEnemy,
    queued::DamageAllEnemy,
    queued::AttackPlayer,
    queued::DamagePlayer,
    queued::VampireAttack,
    queued::PlayerLoseHp,
    queued::HealPlayer,
    queued::MonsterGainBlock,
    queued::RollMove,
    queued::ReactiveRollMove,
    queued::NoOpRollMove,
    queued::ChangeStance,
    queued::GainEnergy,
    queued::GainBlock,
    queued::DrawCards,
    queued::EmptyDeckShuffle,
    queued::ShuffleDrawPile,
    queued::ShuffleTempCardIntoDrawPile,
    queued::PlayTopCard,
    queued::MakeTempCardInHand,
    queued::MakeTempCardInDrawPile,
    queued::MakeTempCardInDiscard,
    queued::MakeTempCardsInHand,
    queued::DiscardNoTriggerCard,
    queued::ClearCardQueue,
    queued::DiscardAtEndOfTurn,
    queued::DiscardAtEndOfTurnHelper,
    queued::RestoreRetainedCards,
    queued::UnnamedEndOfTurnAction,
    queued::MonsterStartTurnAction,
    queued::TriggerEndOfTurnOrbsAction,
    queued::ChannelOrb,
    queued::EvokeFrontOrb,
    queued::ExhaustTopCardInHand,
    queued::ExhaustSpecificCardInHand,
    queued::DamageRandomEnemy,
    queued::ExhaustRandomCardInHand,
    queued::MadnessAction,
    queued::RandomizeHandCost,
    queued::GainBlockRandomEnemy,
    queued::SummonGremlins,
    queued::SpawnTorchHeads,
    queued::SpireShieldDebuff,
    queued::OnAfterCardUsed,
    queued::PutRandomCardsInDrawPile,
    queued::DiscoveryAction,
    queued::InfernalBladeAction,
    queued::JackOfAllTradesAction,
    queued::TransmutationAction,
    queued::ViolenceAction,
    queued::BetterDiscardPileToHandAction,
    queued::ArmamentsAction,
    queued::DualWieldAction,
    queued::ExhumeAction,
    queued::ForethoughtAction,
    queued::HeadbuttAction,
    queued::ChooseExhaustOne,
    queued::DrawToHandAction,
    queued::WarcryAction,
    queued::TimeEaterPlayCardQueueItem,
    queued::UpgradeAllCardsInHand,
    queued::EssenceOfDarkness,
    queued::MakeRandomCardInHand,
    queued::MakeRandomCommonCardInHand,
    queued::DiscardOne,
    queued::RecycleAction,
    queued::ScrapeAction,
    queued::RebootAction,
    queued::SeekAction,
    queued::AllForOneAction,
    queued::IncreaseOrbSlots,
    queued::SuicideAction,
    queued::PoisonLoseHpAction,
    queued::RemovePlayerDebuffs,
    queued::UpgradeRandomCardAction,
    queued::CodexAction,
    queued::ExhaustMany,
    queued::GambleAction,
    queued::ToolboxAction,
    queued::DualityAction,
    queued::ApotheosisAction,
    queued::DropkickAction,
    queued::EnlightenmentAction,
    queued::EntrenchAction,
    queued::FeedAction,
    queued::FiendFireAction,
    queued::HandOfGreedAction,
    queued::LimitBreakAction,
    queued::ReaperAction,
    queued::RitualDaggerAction,
    queued::SecondWindAction,
    queued::SeverSoulExhaustAction,
    queued::SwordBoomerangAction,
    queued::SpotWeaknessAction,
    queued::ScryAction,
    queued::BouncingFlaskAction,
    queued::CalculatedGambleAction,
    queued::EscapePlanAction,
    queued::RandomDiscardAction,
    queued::SetupAction,
    queued::SunderAction,
    queued::ModifyCurrentCardCost,
    queued::StormOfSteelAction,
    queued::BulletTimeAction,
    queued::WhirlwindAction,
    queued::AttackAllMonsterRecursive,
    queued::Judgment,
    queued::Catalyst,
    queued::BuffPlayer,
    queued::DebuffPlayer,
    queued::DecrementStatus,
    queued::RemoveStatus,
    queued::BuffEnemy,
    queued::DebuffEnemy,
    queued::DebuffAllEnemy>;

struct Action {
    ActionPayload payload;
    bool clearOnCombatVictory = true;

    Action() = default;
    template <class Effect> Action(Effect effect, bool clear = true)
        : payload(std::move(effect)), clearOnCombatVictory(clear) {}
    void execute(BattleContext&) const;

    template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
        visit(s.payload); visit(s.clearOnCombatVictory);
    }
};
} // namespace sts
