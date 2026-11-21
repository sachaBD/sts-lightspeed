//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_ML_ACTIONS_H
#define STS_ML_ACTIONS_H

#include <utility>
#include <cstdint>

#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Cards.h"
#include "constants/Potions.h"

#include "combat/InputState.h"
#include "combat/CardSelectInfo.h"

#include "combat/CardPlay.h"

namespace sts {

    struct BattleContext;
    struct CardInstance;
    struct CardPlay;

    typedef std::array<std::uint16_t,5> DamageMatrix;

    struct Actions {
        static void SetState(BattleContext& bc, InputState state);

        // implemented in BattleContext.h because of template
        template <PlayerStatus s> static void BuffPlayer(BattleContext& bc, int amount=1);
        template <PlayerStatus s> static void DebuffPlayer(BattleContext& bc, int amount=1, bool isSourceMonster=true);

        template <PlayerStatus s> static void DecrementStatus(BattleContext& bc, int amount=1);
        template <PlayerStatus s> static void RemoveStatus(BattleContext& bc);

        template <MonsterStatus> static void BuffEnemy(BattleContext& bc, int idx, int amount=1);
        template <MonsterStatus> static void DebuffEnemy(BattleContext& bc, int idx, int amount, bool isSourceMonster=true);
        template <MonsterStatus> static void DebuffAllEnemy(BattleContext& bc, int amount, bool isSourceMonster=true);


        // todo, action and damage actions should clear certain actions from the queue if the player kills all enemies
        static void AttackEnemy(BattleContext& bc, int idx, int damage);

        // for cards that do one hit, like cleave
        static void AttackAllEnemy(BattleContext& bc, int baseDamage);

        // for cards like whirlwind, where damage is calculated once
        static void AttackAllEnemy(BattleContext& bc, DamageMatrix damageMatrix);

        static void DamageEnemy(BattleContext& bc, int idx, int damage);

        // all usages of this in game have pureDamage set to true
        static void DamageAllEnemy(BattleContext& bc, int damage); // todo maybe need to add boolean isPlayerOwner

        static void AttackPlayer(BattleContext& bc, int idx, int damage);
        static void DamagePlayer(BattleContext& bc, int damage, bool selfDamage=false);
        
        static void VampireAttack(BattleContext& bc, int damage); // only used by shelled parasite
        
        static void PlayerLoseHp(BattleContext& bc, int hp, bool selfDamage=false);
        static void HealPlayer(BattleContext& bc, int amount);

        static void MonsterGainBlock(BattleContext& bc, int idx, int amount);
        static void RollMove(BattleContext& bc, int monsterIdx);
        static void ReactiveRollMove(BattleContext& bc); // for writhing mass

        static void NoOpRollMove(BattleContext& bc);

        static void ChangeStance(BattleContext& bc, Stance stance);
        static void GainEnergy(BattleContext& bc, int amount);
        static void GainBlock(BattleContext& bc, int amount);
        static void DrawCards(BattleContext& bc, int amount);


        // the draw pile doesn't actually have to be empty
        // the game calls on shuffle when this is created
        static void EmptyDeckShuffle(BattleContext& bc);

        // used by Deep Breath and Reboot, does not trigger onShuffle relics
        static void ShuffleDrawPile(BattleContext& bc);

        static void ShuffleTempCardIntoDrawPile(BattleContext& bc, CardId id, int count=1);

        static void PlayTopCard(BattleContext& bc, int monsterTargetIdx, bool exhausts);
        static void MakeTempCardInHand(BattleContext& bc, CardId card, bool upgraded= false, int amount = 1);
        static void MakeTempCardInHand(BattleContext& bc, CardInstance c, int amount = 1);
        static void MakeTempCardInDrawPile(BattleContext& bc, const CardInstance &c, int amount, bool shuffleInto);
        static void MakeTempCardInDiscard(BattleContext& bc, const CardInstance &c, int amount = 1);
        static void MakeTempCardsInHand(BattleContext& bc, std::vector<CardInstance> cards);

        static void DiscardNoTriggerCard(BattleContext& bc); // for doubt, shame, etc, discards the BattleContext.curCard



        static void TriggerEndOfTurnOrbsAction(BattleContext& bc);
        static void ChannelOrb(BattleContext& bc, Orb orb, int orbData = 0);
        static void EvokeFrontOrb(BattleContext& bc);

        static void ExhaustTopCardInHand(BattleContext& bc);
        static void ExhaustSpecificCardInHand(BattleContext& bc, int idx, std::int16_t uniqueId);

        // Random Actions

        static void DamageRandomEnemy(BattleContext& bc, int damage); // juggernaut
        static void GainBlockRandomEnemy(BattleContext& bc, int sourceMonster, int amount);

        static void SummonGremlins(BattleContext& bc);
        static void SpawnTorchHeads(BattleContext& bc);
        static void SpireShieldDebuff(BattleContext& bc); // only called if player has orb slots

        static void ExhaustRandomCardInHand(BattleContext& bc, int count);
        static void MadnessAction(BattleContext& bc);
        static void RandomizeHandCost(BattleContext& bc);
        static void UpgradeRandomCardAction(BattleContext& bc); // Warped Tongs Relic

        static void CodexAction(BattleContext& bc); // Nilrys Codex onPlayerEndTurn
        static void ExhaustMany(BattleContext& bc, int limit);
        static void GambleAction(BattleContext& bc);
        static void ToolboxAction(BattleContext& bc);
        static void FiendFireAction(BattleContext& bc, int targetIdx, int calculatedDamage); // Fiend Fire Card
        static void SwordBoomerangAction(BattleContext& bc, int baseDamage); // Sword Boomerang

        static void PutRandomCardsInDrawPile(BattleContext& bc, CardType type, int count); // used by chrysalis and metamorphosis
        static void MakeRandomCardInHand(BattleContext& bc, CardType type, int count, bool zeroCost=false);
        static void MakeRandomCommonCardInHand(BattleContext& bc, int count);
        static void DiscoveryAction(BattleContext& bc, CardType type, int amount); // attack potion, skill potion
        static void InfernalBladeAction(BattleContext& bc);
        static void JackOfAllTradesAction(BattleContext& bc, bool upgraded);
        static void TransmutationAction(BattleContext& bc, bool upgraded, int energy, bool useEnergy);
        static void ViolenceAction(BattleContext& bc, int count);

        static void BetterDiscardPileToHandAction(BattleContext& bc, int amount, CardSelectTask task); // used by hologram and liquid memories
        static void ArmamentsAction(BattleContext& bc);
        static void DualWieldAction(BattleContext& bc, int copyCount);
        static void ExhumeAction(BattleContext& bc);
        static void ForethoughtAction(BattleContext& bc, bool upgraded);
        static void HeadbuttAction(BattleContext& bc);
        static void ChooseExhaustOne(BattleContext& bc);
        static void DiscardOne(BattleContext& bc);
        static void RecycleAction(BattleContext& bc);
        static void ScrapeAction(BattleContext& bc, int drawCount);
        static void RebootAction(BattleContext& bc, int drawCount);
        static void SeekAction(BattleContext& bc, int count);
        static void ScryAction(BattleContext& bc, int count);
        static void AllForOneAction(BattleContext& bc);
        static void DrawToHandAction(BattleContext& bc, CardSelectTask task, CardType cardType);
        static void WarcryAction(BattleContext& bc);

        // ************

        static void UpgradeAllCardsInHand(BattleContext& bc);
        static void EssenceOfDarkness(BattleContext& bc, int darkOrbsPerSlot); // handle sacred bark
        static void IncreaseOrbSlots(BattleContext& bc, int count);

        static void SuicideAction(BattleContext& bc, int monsterIdx, bool triggerRelics);

        static void PoisonLoseHpAction(BattleContext& bc, int monsterIdx);
        static void RemovePlayerDebuffs(BattleContext& bc); // Orange Pellets Relic

        static void DualityAction(BattleContext& bc); // Duality Relic

        static void ApotheosisAction(BattleContext& bc);
        static void DropkickAction(BattleContext& bc, int targetIdx);
        static void EnlightenmentAction(BattleContext& bc, bool upgraded);
        static void EntrenchAction(BattleContext& bc);
        static void FeedAction(BattleContext& bc, int idx, int damage, bool upgraded);
        static void HandOfGreedAction(BattleContext& bc, int idx, int damage, bool upgraded);
        static void LimitBreakAction(BattleContext& bc);
        static void ReaperAction(BattleContext& bc, int baseDamage);
        static void RitualDaggerAction(BattleContext& bc, int idx, int damage);
        static void SecondWindAction(BattleContext& bc, int blockPerCard);
        static void SeverSoulExhaustAction(BattleContext& bc);
        static void SpotWeaknessAction(BattleContext& bc, int target, int strength); // Spot Weakness
        static void SunderAction(BattleContext& bc, int target, int damage);
        static void ModifyCurrentCardCost(BattleContext& bc, int amount);
        static void StormOfSteelAction(BattleContext& bc, bool upgraded);
        static void BulletTimeAction(BattleContext& bc);
        static void BouncingFlaskAction(BattleContext& bc, int poison);
        static void CalculatedGambleAction(BattleContext& bc);
        static void EscapePlanAction(BattleContext& bc, int block);
        static void RandomDiscardAction(BattleContext& bc);
        static void SetupAction(BattleContext& bc);
        static void WhirlwindAction(BattleContext& bc, int baseDamage, int energy, bool useEnergy);

        static void AttackAllMonsterRecursive(BattleContext& bc, DamageMatrix matrix, int timesRemaining);
    };

}

#endif //STS_ML_ACTIONS_H
