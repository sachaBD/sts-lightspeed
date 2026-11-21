//
// Created by gamerpuppy on 7/4/2021.
//

#include "combat/Actions.h"
#include "combat/BattleContext.h"
#include "game/Game.h"
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sts {

namespace {
// Status effects keep their existing template implementations; the queue stores only the enum value.
template <class Status, class F, std::size_t... I>
void visit_status(Status status, F&& f, std::index_sequence<I...>) {
    using Call = void (*)(F&);
    static constexpr Call calls[] = {
        +[](F& call) { call(std::integral_constant<Status, static_cast<Status>(I)>{}); }...
    };
    const auto index = static_cast<std::size_t>(status);
    if (index >= sizeof...(I)) throw std::invalid_argument{"invalid queued status"};
    calls[index](f);
}
} // namespace


Action Actions::SetState(InputState state) {
    return {queued::SetState{state}};
}

void queued::SetState::execute(BattleContext& bc) const {
    bc.setState(state);
}

Action Actions::AttackEnemy(int idx, int damage) {
    return {queued::AttackEnemy{idx, damage}};
}

void queued::AttackEnemy::execute(BattleContext& bc) const {
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].attacked(bc, damage);
    if (bc.player.hasStatus<PS::ENVENOM>()) {
        bc.monsters.arr[idx].buff<MS::POISON>(bc.player.getStatus<PS::ENVENOM>());
    }
    bc.checkCombat();
}

Action Actions::AttackAllEnemy(int baseDamage) {
    return {queued::AttackAllEnemy{baseDamage}};
}

void queued::AttackAllEnemy::execute(BattleContext& bc) const {
    // assume bc.curCard is the card being used

    int damageMatrix[5];
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            damageMatrix[i] = bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage);
        }
    }

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            bc.monsters.arr[i].attacked(bc, damageMatrix[i]);
            if (bc.player.hasStatus<PS::ENVENOM>()) {
                bc.monsters.arr[i].buff<MS::POISON>(bc.player.getStatus<PS::ENVENOM>());
            }
        }
    }

    bc.checkCombat();
}

Action Actions::AttackAllEnemy(DamageMatrix damageMatrix) {
    return {queued::AttackAllEnemyMatrix{damageMatrix}};
}

void queued::AttackAllEnemyMatrix::execute(BattleContext& bc) const {
    // assume bc.curCard is the card being used

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            bc.monsters.arr[i].attacked(bc, static_cast<int>(damageMatrix[i]));
            if (bc.player.hasStatus<PS::ENVENOM>()) {
                bc.monsters.arr[i].buff<MS::POISON>(bc.player.getStatus<PS::ENVENOM>());
            }
        }
    }

    bc.checkCombat();
}

Action Actions::DamageEnemy(int idx, int damage) {
    return {queued::DamageEnemy{idx, damage}};
}

void queued::DamageEnemy::execute(BattleContext& bc) const {
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

Action Actions::DamageAllEnemy(int damage) {
    return {queued::DamageAllEnemy{damage}};
}

void queued::DamageAllEnemy::execute(BattleContext& bc) const {
    for (int idx = 0; idx < bc.monsters.monsterCount; idx++) {
        if (!bc.monsters.arr[idx].isDeadOrEscaped()) {
            bc.monsters.arr[idx].damage(bc, damage); // possible should addToBot here todo
        }
    }
    bc.checkCombat();
}

Action Actions::AttackPlayer(int idx, int damage) {
    return {queued::AttackPlayer{idx, damage}, false};
}

void queued::AttackPlayer::execute(BattleContext& bc) const {
    bc.player.attacked(bc, idx, damage);
}

Action Actions::DamagePlayer(int damage, bool selfDamage) {
    return {queued::DamagePlayer{damage, selfDamage}, false};
}

void queued::DamagePlayer::execute(BattleContext& bc) const {
    bc.player.damage(bc, damage, selfDamage);
}

Action Actions::VampireAttack(int damage) {
    return {queued::VampireAttack{damage}};
}

void queued::VampireAttack::execute(BattleContext& bc) const {
    const auto mIdx = 0;
    auto &m = bc.monsters.arr[mIdx]; // only used by shelled parasite so idx is 0
    bc.player.attacked(bc, mIdx, damage);
    if (m.isAlive()) {
        m.heal(std::min(damage, static_cast<int>(bc.player.lastAttackUnblockedDamage)));
    }
}

Action Actions::PlayerLoseHp(int hp, bool selfDamage) {
    return {queued::PlayerLoseHp{hp, selfDamage}, false};
}

void queued::PlayerLoseHp::execute(BattleContext& bc) const {
    bc.player.loseHp(bc, hp, selfDamage);
}

Action Actions::HealPlayer(int amount) {
    return {queued::HealPlayer{amount}, false};
}

void queued::HealPlayer::execute(BattleContext& bc) const {
    bc.player.heal(amount);
}

Action Actions::MonsterGainBlock(int idx, int amount) {
    return {queued::MonsterGainBlock{idx, amount}};
}

void queued::MonsterGainBlock::execute(BattleContext& bc) const {
    bc.monsters.arr[idx].block += amount;
}

Action Actions::RollMove(int monsterIdx) {
    return {queued::RollMove{monsterIdx}};
}

void queued::RollMove::execute(BattleContext& bc) const {
    Monster &m = bc.monsters.arr[monsterIdx];
    m.rollMove(bc);
}

Action Actions::ReactiveRollMove() {
    return {queued::ReactiveRollMove{}};
}

void queued::ReactiveRollMove::execute(BattleContext& bc) const {
    // writhing mass is always monster 0
    Monster &m = bc.monsters.arr[0];

    for (int i = 0 ; i < m.getStatus<MS::REACTIVE>(); ++i) {
        m.rollMove(bc);
    }
    m.setStatus<MS::REACTIVE>(0);
}

Action Actions::NoOpRollMove() {
    return {queued::NoOpRollMove{}};
}

void queued::NoOpRollMove::execute(BattleContext& bc) const {
    bc.noOpRollMove();
}

Action Actions::ChangeStance(Stance stance) {
    return {queued::ChangeStance{stance}};
}

void queued::ChangeStance::execute(BattleContext& bc) const {
    const auto previousStance = bc.player.stance;
    if (previousStance == stance) {
        return;
    }

    // These are the player-side transition effects required by the
    // current simulator's Calm/Wrath/Divinity implementation.
    if (previousStance == Stance::CALM) {
        bc.player.gainEnergy(2);
    }
    if (previousStance != Stance::NEUTRAL && bc.player.nirvanaBlock > 0) {
        bc.addToBot(Actions::GainBlock(bc.player.nirvanaBlock));
    }
    if (bc.player.mentalFortressBlock > 0) {
        bc.addToBot(Actions::GainBlock(bc.player.mentalFortressBlock));
    }
    bc.player.stance = stance;
    if (stance == Stance::WRATH && bc.player.rushdownDraw > 0) {
        bc.addToBot(Actions::DrawCards(bc.player.rushdownDraw));
    }
    if (stance == Stance::DIVINITY) {
        bc.player.gainEnergy(3);
    }
}

Action Actions::GainEnergy(int amount) {
    return {queued::GainEnergy{amount}};
}

void queued::GainEnergy::execute(BattleContext& bc) const {
    bc.player.gainEnergy(amount);
}

Action Actions::GainBlock(int amount) {
    return {queued::GainBlock{amount}, false};
}

void queued::GainBlock::execute(BattleContext& bc) const {
    bc.player.gainBlock(bc, amount);
}
Action Actions::DrawCards(int amount) {
    return {queued::DrawCards{amount}};
}

void queued::DrawCards::execute(BattleContext& bc) const {
    bc.drawCards(amount);
}

Action Actions::EmptyDeckShuffle() {
    return {queued::EmptyDeckShuffle{}};
}

void queued::EmptyDeckShuffle::execute(BattleContext& bc) const {
    java::Collections::shuffle(
            bc.cards.discardPile.begin(),
            bc.cards.discardPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );

    bc.cards.moveDiscardPileIntoToDrawPile();
}

Action Actions::ShuffleDrawPile() {
    return {queued::ShuffleDrawPile{}};
}

void queued::ShuffleDrawPile::execute(BattleContext& bc) const {
    java::Collections::shuffle(
            bc.cards.drawPile.begin(),
            bc.cards.drawPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );
}

Action Actions::ShuffleTempCardIntoDrawPile(CardId id, int count) {
    return {queued::ShuffleTempCardIntoDrawPile{id, count}};
}

void queued::ShuffleTempCardIntoDrawPile::execute(BattleContext& bc) const {
    CardInstance c(id);
    for (int i = 0; i < count; ++i) {
        const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
        bc.cards.createTempCardInDrawPile(idx, c);
    }
}

Action Actions::PlayTopCard(int monsterTargetIdx, bool exhausts) {
    return {queued::PlayTopCard{monsterTargetIdx, exhausts}};
}

void queued::PlayTopCard::execute(BattleContext& bc) const {
    bc.playTopCardInDrawPile(monsterTargetIdx, exhausts);
}


// todo fix the arguments are used wrongly all over ? what did this mean
// todo check for master reality
Action Actions::MakeTempCardInHand(CardId card, bool upgraded, int amount) {
    CardInstance c(card, upgraded);
    return Actions::MakeTempCardInHand(c, amount);
}

Action Actions::MakeTempCardInHand(CardInstance card, int amount) {
    return {queued::MakeTempCardInHand{card, amount}};
}

void queued::MakeTempCardInHand::execute(BattleContext& bc) const {
    for (int i = 0; i < amount; ++i) {
        CardInstance c(card);
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}

Action Actions::MakeTempCardInDrawPile(const CardInstance &c, int amount, bool shuffleInto) {
    return {queued::MakeTempCardInDrawPile{c, amount, shuffleInto}};
}

void queued::MakeTempCardInDrawPile::execute(BattleContext& bc) const {
    for (int i = 0; i < amount; ++i) {
        if (shuffleInto) {
            const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
            bc.cards.createTempCardInDrawPile(idx, c);
        }
        // todo else
    }
}

Action Actions::MakeTempCardInDiscard(const CardInstance &c, int amount) {
    return {queued::MakeTempCardInDiscard{c, amount}};
}

void queued::MakeTempCardInDiscard::execute(BattleContext& bc) const {
    for (int i = 0; i < amount; ++i) {
        bc.cards.createTempCardInDiscard(c);
    }
}

Action Actions::MakeTempCardsInHand(std::vector<CardInstance> cards) {
    return {queued::MakeTempCardsInHand{std::move(cards)}};
}

void queued::MakeTempCardsInHand::execute(BattleContext& bc) const {
    for (auto c : cards) {
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}


Action Actions::DiscardNoTriggerCard() {
    return {queued::DiscardNoTriggerCard{}};
}

void queued::DiscardNoTriggerCard::execute(BattleContext& bc) const {
    const auto &c = bc.curCardQueueItem.card;
    bc.cards.notifyRemoveFromHand(c);
    bc.cards.moveToDiscardPile(c);
}

Action Actions::ClearCardQueue() {
    return {queued::ClearCardQueue{}};
}

void queued::ClearCardQueue::execute(BattleContext& bc) const {
    bc.cardQueue.clear();
}

Action Actions::DiscardAtEndOfTurn() {
    return {queued::DiscardAtEndOfTurn{}};
}

void queued::DiscardAtEndOfTurn::execute(BattleContext& bc) const {
    bc.discardAtEndOfTurn();
}

Action Actions::DiscardAtEndOfTurnHelper() {
    return {queued::DiscardAtEndOfTurnHelper{}};
}

void queued::DiscardAtEndOfTurnHelper::execute(BattleContext& bc) const {
    bc.discardAtEndOfTurnHelper();
}

Action Actions::RestoreRetainedCards(int count) {
    return {queued::RestoreRetainedCards{count}};
}

void queued::RestoreRetainedCards::execute(BattleContext& bc) const {
    bc.restoreRetainedCards(count);
}

Action Actions::UnnamedEndOfTurnAction() {
    return {queued::UnnamedEndOfTurnAction{}};
}

void queued::UnnamedEndOfTurnAction::execute(BattleContext& bc) const {
    bc.turnHasEnded = true;
    if (!bc.skipMonsterTurn) {
        bc.addToBot(Actions::MonsterStartTurnAction());
        bc.monsterTurnIdx = 0; // monstergroup preincrements this
    }
}

Action Actions::MonsterStartTurnAction() {
    return {queued::MonsterStartTurnAction{}};
}

void queued::MonsterStartTurnAction::execute(BattleContext& bc) const {
    bc.monsters.applyPreTurnLogic(bc);
}

Action Actions::TriggerEndOfTurnOrbsAction() {
    return {queued::TriggerEndOfTurnOrbsAction{}};
}

void queued::TriggerEndOfTurnOrbsAction::execute(BattleContext& bc) const {
    bc.player.triggerEndOfTurnOrbs(bc);
}

Action Actions::ChannelOrb(Orb orb, int orbData) {
    return {queued::ChannelOrb{orb, orbData}};
}

void queued::ChannelOrb::execute(BattleContext& bc) const {
    bc.player.channelOrb(bc, orb, orbData);
}

Action Actions::EvokeFrontOrb() {
    return {queued::EvokeFrontOrb{}};
}

void queued::EvokeFrontOrb::execute(BattleContext& bc) const {
    bc.player.evokeFrontOrb(bc);
}

Action Actions::ExhaustTopCardInHand() {
    return {queued::ExhaustTopCardInHand{}};
}

void queued::ExhaustTopCardInHand::execute(BattleContext& bc) const {
    bc.exhaustTopCardInHand();
}

Action Actions::ExhaustSpecificCardInHand(int idx, std::int16_t uniqueId) {
    return {queued::ExhaustSpecificCardInHand{idx, uniqueId}};
}

void queued::ExhaustSpecificCardInHand::execute(BattleContext& bc) const {
    bc.exhaustSpecificCardInHand(idx, uniqueId);
}

Action Actions::DamageRandomEnemy(int damage) {
    return {queued::DamageRandomEnemy{damage}};
}

void queued::DamageRandomEnemy::execute(BattleContext& bc) const {
    const int idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }
    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

Action Actions::ExhaustRandomCardInHand(int count) {
    return {queued::ExhaustRandomCardInHand{count}};
}

void queued::ExhaustRandomCardInHand::execute(BattleContext& bc) const {
    for (int i = 0; i < count; ++i) {
        if (bc.cards.cardsInHand <= 0) {
            return;
        }
        const auto idx = bc.cards.getRandomCardIdxInHand(bc.cardRandomRng);
        auto c = bc.cards.hand[idx];
        bc.cards.removeFromHandAtIdx(idx);
        bc.triggerAndMoveToExhaustPile(c);
    }
}

Action Actions::MadnessAction() {
    return {queued::MadnessAction{}};
}

void queued::MadnessAction::execute(BattleContext& bc) const {
            bool haveNonZeroCost = false;
            bool haveNonZeroTurnCost = false;
            for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                const auto &c = bc.cards.hand[i];

                if (c.costForTurn > 0) {
                    haveNonZeroTurnCost = true;
                    break;
                }

                if (c.cost > 0) {
                    haveNonZeroCost = true;
                }
            }

            const auto haveValidCard = haveNonZeroCost || haveNonZeroTurnCost;
            if (!haveValidCard) {
                return;
            }

            // always have 1 or more cards in hand here
    //#pragma clang diagnostic push
    //#pragma ide diagnostic ignored "EndlessLoop"
            while (true) {
                const auto randomIdx = bc.cardRandomRng.random(bc.cards.cardsInHand-1);
                auto &c = bc.cards.hand[randomIdx];

                if (haveNonZeroTurnCost) {
                    if (c.costForTurn > 0) {
                        c.cost = 0;
                        c.costForTurn = 0;
                        break;
                    } else {
                        continue;
                    }

                } else {
                    if (c.cost > 0) {
                        c.cost = 0;
                        c.costForTurn = 0;
                        break;

                    } else {
                        continue;
                    }
                }
            }
    //#pragma clang diagnostic pop
}

Action Actions::RandomizeHandCost() {
    return {queued::RandomizeHandCost{}};
}

void queued::RandomizeHandCost::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.cost >= 0) {
            int newCost = bc.cardRandomRng.random(3);
            c.cost = newCost;
            c.costForTurn = newCost;
        }
    }
}

Action Actions::GainBlockRandomEnemy(int sourceMonster, int amount) {
    return {queued::GainBlockRandomEnemy{sourceMonster, amount}};
}

void queued::GainBlockRandomEnemy::execute(BattleContext& bc) const {
    int validIdxs[5];
    int validCount = 0;

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &m = bc.monsters.arr[i];
        if (i != sourceMonster && !m.isDying()) {
            validIdxs[validCount++] = i;
        }
    }

    int targetIdx;
    if (validCount > 0) {
        targetIdx = validIdxs[bc.aiRng.random(validCount - 1)];
    } else {
        targetIdx = sourceMonster;
    }

    bc.monsters.arr[targetIdx].addBlock(amount);
}

Action Actions::SummonGremlins() {
    return {queued::SummonGremlins{}};
}

void queued::SummonGremlins::execute(BattleContext& bc) const {
            int openIdxCount = 0;
            int newGremlinIdxs[2];
            if (bc.monsters.arr[1].isDying()) {
                newGremlinIdxs[openIdxCount++] = 1;
            }
            if (bc.monsters.arr[2].isDying()) {
                newGremlinIdxs[openIdxCount++] = 2;
            }
            if (openIdxCount < 2 && bc.monsters.arr[0].isDying()) {
                newGremlinIdxs[openIdxCount++] = 0;
            }
    #ifdef sts_asserts
            assert(openIdxCount == 2);
    #endif

            auto &gremlin0 = bc.monsters.arr[newGremlinIdxs[0]];
            auto &gremlin1 = bc.monsters.arr[newGremlinIdxs[1]];

            gremlin0 = Monster();
            gremlin1 = Monster();

            gremlin0.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[0]);
            gremlin1.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[1]);
            bc.monsters.monstersAlive += 2;

            if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
                gremlin0.buff<MS::STRENGTH>(1);
                gremlin1.buff<MS::STRENGTH>(1);
            }
            gremlin0.buff<MS::MINION>();
            gremlin1.buff<MS::MINION>();

            gremlin0.rollMove(bc);
            gremlin1.rollMove(bc);
}

Action Actions::SpawnTorchHeads() {
    return {queued::SpawnTorchHeads{}};
}

void queued::SpawnTorchHeads::execute(BattleContext& bc) const {
            const auto spawnCount = 3-bc.monsters.monstersAlive;
    #ifdef sts_asserts
            assert(spawnCount > 0);
    #endif
            const int spawnIdxs[2] {(bc.monsters.arr[1].isDying() ? 1 : 0), 0};

            for (int i = 0; i < spawnCount; ++i) {
                const auto idx = spawnIdxs[i];
                auto &torchHead = bc.monsters.arr[idx];
                torchHead = Monster();
                torchHead.construct(bc, MonsterId::TORCH_HEAD, idx);
                torchHead.initHp(bc.monsterHpRng, bc.ascension); // bug somewhere in game
                torchHead.setMove(MMID::TORCH_HEAD_TACKLE);
                torchHead.buff<MS::MINION>();

                if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
                    torchHead.buff<MS::STRENGTH>(1);
                }
                ++bc.monsters.monstersAlive;
            }

            for (int i = 0; i < spawnCount; ++i) {
                bc.noOpRollMove();
            }
}

Action Actions::SpireShieldDebuff() {
    return {queued::SpireShieldDebuff{}};
}

void queued::SpireShieldDebuff::execute(BattleContext& bc) const {
    if (bc.aiRng.randomBoolean()) {
        bc.player.debuff<PS::FOCUS>(-1);
    } else {
        bc.player.debuff<PS::STRENGTH>(-1);
    }
}


Action Actions::OnAfterCardUsed() {
    return {queued::OnAfterCardUsed{}, false};
}

void queued::OnAfterCardUsed::execute(BattleContext& bc) const {
    bc.onAfterUseCard();
}

Action Actions::PutRandomCardsInDrawPile(CardType type, int count) {
    return {queued::PutRandomCardsInDrawPile{type, count}};
}

void queued::PutRandomCardsInDrawPile::execute(BattleContext& bc) const {
    CardId ids[5];
    for (int i = 0; i < count; ++i) {
        ids[i] = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, type);
    }

    for (int i = 0; i < count; ++i) {
        CardInstance card(ids[i], false);
        card.cost = 0;
        card.costForTurn = 0;

        const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
        bc.cards.createTempCardInDrawPile(idx, card);
    }
}

Action Actions::DiscoveryAction(CardType type, int amount) {
    return {queued::DiscoveryAction{type, amount}};
}

void queued::DiscoveryAction::execute(BattleContext& bc) const {
    bc.haveUsedDiscoveryAction = true;
    bc.openDiscoveryScreen(sts::generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, type), amount);
}

Action Actions::InfernalBladeAction() {
    return {queued::InfernalBladeAction{}};
}

void queued::InfernalBladeAction::execute(BattleContext& bc) const {
    const auto cid = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, CardType::ATTACK);
    CardInstance c(cid);
    c.setCostForTurn(0);
    bc.addToTop( Actions::MakeTempCardInHand(c) );
}

Action Actions::JackOfAllTradesAction(bool upgraded) {
    return {queued::JackOfAllTradesAction{upgraded}};
}

void queued::JackOfAllTradesAction::execute(BattleContext& bc) const {
    const auto c1 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
    bc.addToTop( Actions::MakeTempCardInHand(c1) );
    if (upgraded) {
        auto c2 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
        bc.addToTop( Actions::MakeTempCardInHand(c2) );
    }
}

Action Actions::TransmutationAction(bool upgraded, int energy, bool useEnergy) {
    return {queued::TransmutationAction{upgraded, energy, useEnergy}};
}

void queued::TransmutationAction::execute(BattleContext& bc) const {
     const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);

     if (effectAmount == 0) {
         return;
     }

     std::vector<CardInstance> cards;
    for (int i = 0; i < effectAmount; ++i) {
        const auto cid = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
        CardInstance c(cid, upgraded);
        c.setCostForTurn(0);
        cards.push_back(c);
    }
    bc.addToBot( Actions::MakeTempCardsInHand(cards) );

    if (useEnergy) {
        bc.player.useEnergy(bc.player.energy);
    }
}

Action Actions::ViolenceAction(int count) {
    return {queued::ViolenceAction{count}};
}

void queued::ViolenceAction::execute(BattleContext& bc) const {
    fixed_list<int,CardManager::MAX_GROUP_SIZE> attackIdxList;
    for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
        const auto &c = bc.cards.drawPile[i];
        if (c.getType() == CardType::ATTACK) {

            if (attackIdxList.empty()) {
                attackIdxList.push_back(i);
            } else {
                const auto randomIdx = bc.cardRandomRng.random(attackIdxList.size() - 1);
                attackIdxList.insert(randomIdx, i);
            }
        }
    }

    if (attackIdxList.empty()) {
        return;
    }

    int removeIdxs[4];
    // hack to do this faster: the attackList is just pushed forward by i so we skip removing from bottom
    int i = 0;
    for (; i < count; ++i) {
        if (attackIdxList.size()-i <= 0) {
            return;
        }

        java::Collections::shuffle(attackIdxList.begin()+i, attackIdxList.end(), java::Random(bc.shuffleRng.randomLong()));
        const auto removeIdx = attackIdxList[i];
        removeIdxs[i] = removeIdx;

        const auto &c = bc.cards.drawPile[removeIdx];
        if (bc.cards.cardsInHand == 10) {
            bc.cards.moveToDiscardPile(c);
        } else {
            bc.cards.moveToHand(c);
        }
    }

    std::sort(removeIdxs, removeIdxs+i);
    for (int x = i-1; x >= 0; --x) {
        const auto drawPileRemoveIdx = removeIdxs[x];
        bc.cards.removeFromDrawPileAtIdx(drawPileRemoveIdx);
    }
}

// todo the amount should be the copies put into the hand 2 if have sacred bark and liquid memories
Action Actions::BetterDiscardPileToHandAction(int amount, CardSelectTask task) {
    return {queued::BetterDiscardPileToHandAction{amount, task}};
}

void queued::BetterDiscardPileToHandAction::execute(BattleContext& bc) const {
    if (bc.cards.discardPile.empty()) {
        return;
    }
    if (bc.cards.discardPile.size() == 1) {
        const int idx = 0;
        bc.chooseDiscardToHandCard(0, task==CardSelectTask::LIQUID_MEMORIES_POTION);
    } else {
        bc.openSimpleCardSelectScreen(task, 1);
    }
}

Action Actions::ArmamentsAction() {
    return {queued::ArmamentsAction{}};
}

void queued::ArmamentsAction::execute(BattleContext& bc) const {
    int canUpgradeCount = 0;
    int lastUpgradeIdx = 0;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {
            ++canUpgradeCount;
            lastUpgradeIdx = i;
        }
    }

    if (canUpgradeCount == 0) {
        // do nothing

    } else if (canUpgradeCount == 1) {
        bc.cards.hand[lastUpgradeIdx].upgrade();

    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::ARMAMENTS, 1);
    }
}

Action Actions::DualWieldAction(int copyCount) {
    return {queued::DualWieldAction{copyCount}};
}

void queued::DualWieldAction::execute(BattleContext& bc) const {
    //        fixed_list<CardInstance, 10> validCards;
    //        fixed_list<CardInstance, 10> invalidCards;
    //
    //        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
    //            const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
    //            if (valid) {
    //                validCards.push_back(bc.cards.hand[i]);
    //
    //            } else {
    //                invalidCards.push_back(bc.cards.hand[i]);
    //            }
    //        }

            int validCount = 0;
            int lastValidIdx = 0;

            for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
                if (valid) {
                    ++validCount;
                    lastValidIdx = i;
                }
            }

            if (validCount == 0) {
                return;
            }

            if (validCount == 1) {
                for (int i = 0; i < copyCount; ++i) {
                    auto c = bc.cards.hand[lastValidIdx];
                    if (bc.cards.cardsInHand + 1 <= CardManager::MAX_HAND_SIZE) {
                        bc.cards.createTempCardInHand(c);

                    } else {
                        bc.cards.createTempCardInDiscard(c);

                    }
                }

            } else {
                bc.inputState = InputState::CARD_SELECT;
                bc.cardSelectInfo.cardSelectTask = CardSelectTask::DUAL_WIELD;
                bc.cardSelectInfo.dualWield_CopyCount() = copyCount;

            }
}

Action Actions::ExhumeAction() {
    return {queued::ExhumeAction{}};
}

void queued::ExhumeAction::execute(BattleContext& bc) const {
    if (bc.cards.exhaustPile.empty() || bc.cards.cardsInHand == 10) {
        return;
    }

    int nonExhumeCards = 0;
    int lastNonExhumeIdx = -1;
    for (int i = 0; i < bc.cards.exhaustPile.size(); ++i) {
        if (bc.cards.exhaustPile[i].id != CardId::EXHUME) {
            ++nonExhumeCards;
            lastNonExhumeIdx = i;
        }
    }

    if (nonExhumeCards == 0) {
        return;

    } else if (nonExhumeCards == 1) {
        bc.chooseExhumeCard(lastNonExhumeIdx);

    } else {
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::EXHUME;
        bc.inputState = InputState::CARD_SELECT;
    }
}

Action Actions::ForethoughtAction(bool upgraded) {
    return {queued::ForethoughtAction{upgraded}};
}

void queued::ForethoughtAction::execute(BattleContext& bc) const {
            if (bc.cards.cardsInHand == 0) {
                return;
            }

            // todo implement Upgraded version
    //        //
    //        if (upgraded) {
    //            bc.cardSelectInfo.cardSelectTask = CardSelectTask::FORETHOUGHT;
    //            bc.cardSelectInfo.canPickAnyNumber = true;
    //            bc.inputState = InputState::CARD_SELECT;
    //
    //        } else {
            if (bc.cards.cardsInHand == 1) {
                bc.chooseForethoughtCard(0);
            } else {
                bc.cardSelectInfo.cardSelectTask = CardSelectTask::FORETHOUGHT;
                bc.cardSelectInfo.canPickAnyNumber = false;
                bc.inputState = InputState::CARD_SELECT;
            }
    //        }
}

Action Actions::HeadbuttAction() {
    return {queued::HeadbuttAction{}};
}

void queued::HeadbuttAction::execute(BattleContext& bc) const {
    if (bc.cards.discardPile.empty()) {
        return;

    } else if (bc.cards.discardPile.size() == 1) {
        bc.chooseHeadbuttCard(0);
    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::HEADBUTT, 1);
    }
}

Action Actions::ChooseExhaustOne() {
    return {queued::ChooseExhaustOne{}};
}

void queued::ChooseExhaustOne::execute(BattleContext& bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;

    } else if (bc.cards.cardsInHand == 1) {
        bc.chooseExhaustOneCard(0);

    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::EXHAUST_ONE, 1);

    }
}

Action Actions::DrawToHandAction(CardSelectTask task, CardType cardType) {
    return {queued::DrawToHandAction{task, cardType}};
}

void queued::DrawToHandAction::execute(BattleContext& bc) const {
    int count = 0;
    int idx = 0;

    for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
        const auto &c = bc.cards.drawPile[i];
        if (c.getType() == cardType) {
            if (count > 0) {
                // for keeping rng consistent with game
                // the game creates a temporary list with the skills
                bc.cardRandomRng.random(count - 1);
            }
            idx = i;
            ++count;
        }
    }

    if (count == 0) {
        return;
    }

    if (count == 1) {
        bc.chooseDrawToHandCards(&idx, 1);

    } else {
        bc.cardSelectInfo.cardSelectTask = task;
        bc.inputState = InputState::CARD_SELECT;
    }
}

Action Actions::WarcryAction() {
    return {queued::WarcryAction{}};
}

void queued::WarcryAction::execute(BattleContext& bc) const {
    // todo if the handSize equals or less than the cardsToChoose just choose them here
    if (bc.cards.cardsInHand == 0) {
        return;
    }

    if (bc.cards.cardsInHand == 1) {
        bc.cardRandomRng.random(1);
        bc.chooseWarcryCard(0);

    } else {
        bc.inputState = InputState::CARD_SELECT;
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::WARCRY;
    }
}

Action Actions::TimeEaterPlayCardQueueItem(const CardQueueItem &x) {
    return {queued::TimeEaterPlayCardQueueItem{x}, false};
}

void queued::TimeEaterPlayCardQueueItem::execute(BattleContext& bc) const {
    auto item = x;
    item.exhaustOnUse |= bc.curCardQueueItem.card.doesExhaust();
    item.triggerOnUse = false;
    bc.curCardQueueItem = item;
    bc.onAfterUseCard();
}

Action Actions::UpgradeAllCardsInHand() {
    return {queued::UpgradeAllCardsInHand{}};
}

void queued::UpgradeAllCardsInHand::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {  // curses and statuses can't be upgraded
            bc.cards.hand[i].upgrade();
        }
    }
}

Action Actions::EssenceOfDarkness(int darkOrbsPerSlot) {
    return {queued::EssenceOfDarkness{darkOrbsPerSlot}};
}

void queued::EssenceOfDarkness::execute(BattleContext& bc) const {
    const auto count = bc.player.orbSlots * darkOrbsPerSlot;
    for (int i = 0; i < count; ++i) {
        bc.player.channelOrb(bc, Orb::DARK);
    }
}

Action Actions::MakeRandomCardInHand(CardType type, int count, bool zeroCost) {
    return {queued::MakeRandomCardInHand{type, count, zeroCost}};
}

void queued::MakeRandomCardInHand::execute(BattleContext& bc) const {
    for (int i = 0; i < count; ++i) {
        CardInstance card(getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, type));
        if (zeroCost) {
            card.setCostForTurn(0);
        }
        bc.moveToHandHelper(card);
    }
}

Action Actions::MakeRandomCommonCardInHand(int count) {
    return {queued::MakeRandomCommonCardInHand{count}};
}

void queued::MakeRandomCommonCardInHand::execute(BattleContext& bc) const {
    for (int i = 0; i < count; ++i) {
        CardId id = CardId::INVALID;
        do {
            id = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc);
        } while (getCardRarity(id) != CardRarity::COMMON);
        bc.moveToHandHelper(CardInstance(id));
    }
}

Action Actions::DiscardOne() {
    return {queued::DiscardOne{}};
}

void queued::DiscardOne::execute(BattleContext& bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    if (bc.cards.cardsInHand == 1) {
        bc.chooseDiscardCard(0);
        return;
    }
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::DISCARD_ONE;
    bc.inputState = InputState::CARD_SELECT;
}

Action Actions::RecycleAction() {
    return {queued::RecycleAction{}};
}

void queued::RecycleAction::execute(BattleContext& bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    if (bc.cards.cardsInHand == 1) {
        bc.chooseRecycleCard(0);
        return;
    }
    bc.openSimpleCardSelectScreen(CardSelectTask::RECYCLE, 1);
}

Action Actions::ScrapeAction(int drawCount) {
    return {queued::ScrapeAction{drawCount}};
}

void queued::ScrapeAction::execute(BattleContext& bc) const {
    bc.drawCards(drawCount);
    for (int i = bc.cards.cardsInHand - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        if (card.costForTurn != 0) {
            bc.cards.removeFromHandAtIdx(i);
            bc.cards.moveToDiscardPile(card);
            bc.onManualDiscard(card);
        }
    }
}

Action Actions::RebootAction(int drawCount) {
    return {queued::RebootAction{drawCount}};
}

void queued::RebootAction::execute(BattleContext& bc) const {
    for (int i = bc.cards.cardsInHand - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.shuffleIntoDrawPile(bc.cardRandomRng, card);
    }
    bc.drawCards(drawCount);
}

Action Actions::SeekAction(int count) {
    return {queued::SeekAction{count}};
}

void queued::SeekAction::execute(BattleContext& bc) const {
    if (bc.cards.drawPile.empty() || count <= 0) {
        return;
    }
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::SEEK;
    bc.cardSelectInfo.pickCount = 1;
    bc.cardSelectInfo.canPickAnyNumber = false;
    bc.cardSelectInfo.canPickZero = false;
    bc.cardSelectInfo.data0 = count;
}

Action Actions::AllForOneAction() {
    return {queued::AllForOneAction{}};
}

void queued::AllForOneAction::execute(BattleContext& bc) const {
    // All for One returns every currently-zero-cost discard card to hand;
    // it does not play those cards. Preserve discard-pile iteration order,
    // while removing in reverse to keep indexes stable.
    std::vector<CardInstance> returned;
    for (const auto &card : bc.cards.discardPile) {
        if (card.costForTurn == 0) {
            returned.push_back(card);
        }
    }
    for (int i = static_cast<int>(bc.cards.discardPile.size()) - 1; i >= 0; --i) {
        if (bc.cards.discardPile[i].costForTurn == 0) {
            bc.cards.removeFromDiscard(i);
        }
    }
    for (const auto &card : returned) {
        bc.moveToHandHelper(card);
    }
}

Action Actions::IncreaseOrbSlots(int count) {
    return {queued::IncreaseOrbSlots{count}};
}

void queued::IncreaseOrbSlots::execute(BattleContext& bc) const {
    bc.player.increaseOrbSlots(count);
}

Action Actions::SuicideAction(int monsterIdx, bool triggerRelics) {
    return {queued::SuicideAction{monsterIdx, triggerRelics}};
}

void queued::SuicideAction::execute(BattleContext& bc) const {
    auto &m = bc.monsters.arr[monsterIdx];
    if (triggerRelics) {
        if (m.isAlive()) {
            m.damage(bc, m.curHp);
        }
    } else {
        bc.monsters.arr[monsterIdx].suicideAction(bc);
    }
}

Action Actions::PoisonLoseHpAction(int monsterIdx) {
    return {queued::PoisonLoseHpAction{monsterIdx}};
}

void queued::PoisonLoseHpAction::execute(BattleContext& bc) const {
    auto &monster = bc.monsters.arr[monsterIdx];
    if (monster.isDeadOrEscaped() || !monster.hasStatus<MonsterStatus::POISON>()) {
        return;
    }
    const auto poison = monster.getStatus<MonsterStatus::POISON>();
    monster.damage(bc, poison);
    // Poison reduces after dealing its current value, including when the
    // damage kills the monster. This mirrors the game's start-of-turn
    // ordering and keeps multiple poisoned monsters independent.
    monster.decrementStatus<MonsterStatus::POISON>();
}

Action Actions::RemovePlayerDebuffs() {
    return {queued::RemovePlayerDebuffs{}};
}

void queued::RemovePlayerDebuffs::execute(BattleContext& bc) const {
    bc.player.removeDebuffs();
}

Action Actions::UpgradeRandomCardAction() {
    return {queued::UpgradeRandomCardAction{}};
}

void queued::UpgradeRandomCardAction::execute(BattleContext& bc) const {
    fixed_list<int,10> upgradeableHandIdxs;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {
            upgradeableHandIdxs.push_back(i);
        }
    }

    if (upgradeableHandIdxs.empty()) {
        return;
    }

    java::Collections::shuffle(
            upgradeableHandIdxs.begin(),
            upgradeableHandIdxs.end(),
            java::Random(bc.shuffleRng.randomLong())
    );

    const auto upgradeIdx = upgradeableHandIdxs[0];
    bc.cards.hand[upgradeIdx].upgrade();
}

Action Actions::CodexAction() {
    return {queued::CodexAction{}};
}

void queued::CodexAction::execute(BattleContext& bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::CODEX;
    bc.cardSelectInfo.codexCards() =
            generateDiscoveryCards(bc.cardRandomRng, CharacterClass::IRONCLAD, CardType::INVALID);
}

Action Actions::ExhaustMany(int limit) {
    return {queued::ExhaustMany{limit}};
}

void queued::ExhaustMany::execute(BattleContext& bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::EXHAUST_MANY;
    bc.cardSelectInfo.pickCount = limit;
}

Action Actions::GambleAction() {
    return {queued::GambleAction{}};
}

void queued::GambleAction::execute(BattleContext& bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::GAMBLE;
}

Action Actions::ToolboxAction() {
    return {queued::ToolboxAction{}};
}

void queued::ToolboxAction::execute(BattleContext& bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::DISCOVERY;
    bc.cardSelectInfo.discovery_CopyCount() = 1;
    bc.cardSelectInfo.discovery_Cards() =
            generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, CardType::STATUS); // status is mapped to colorless
}

Action Actions::DualityAction() {
    return {queued::DualityAction{}};
}

void queued::DualityAction::execute(BattleContext& bc) const {
    bc.player.buff<PS::DEXTERITY>(1);
    bc.player.debuff<PS::LOSE_DEXTERITY>(1);
}

Action Actions::ApotheosisAction() {
    return {queued::ApotheosisAction{}};
}

void queued::ApotheosisAction::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.drawPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.discardPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.exhaustPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }
}

Action Actions::DropkickAction(int targetIdx) {
    return {queued::DropkickAction{targetIdx}};
}

void queued::DropkickAction::execute(BattleContext& bc) const {
    if (bc.monsters.arr[targetIdx].isTargetable() && bc.monsters.arr[targetIdx].hasStatus<MS::VULNERABLE>()) {
        bc.addToTop(Actions::DrawCards(1));
        bc.addToTop(Actions::GainEnergy(1));
    }

    const auto &c = bc.curCardQueueItem.card;
    const int damage = bc.calculateCardDamage(c, targetIdx, c.isUpgraded() ? 8 : 5);
    bc.addToTop(Actions::AttackEnemy(targetIdx, damage));
}

Action Actions::EnlightenmentAction(bool upgraded) {
    return {queued::EnlightenmentAction{upgraded}};
}

void queued::EnlightenmentAction::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.costForTurn > 1) {
            c.costForTurn = 1;
        }
        if (upgraded && c.cost > 1) {
            c.costForTurn = 1;
            c.cost = 1;
        }
    }
}

Action Actions::EntrenchAction() {
    return {queued::EntrenchAction{}};
}

void queued::EntrenchAction::execute(BattleContext& bc) const {
    bc.player.gainBlock(bc, bc.player.block);
}

Action Actions::FeedAction(int idx, int damage, bool upgraded) {
    return {queued::FeedAction{idx, damage, upgraded}};
}

void queued::FeedAction::execute(BattleContext& bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].attacked(bc, damage);

    const bool effectTriggered = !m.hasStatus<MS::MINION>()
            && !m.isAlive()
            && !m.isHalfDead()
            && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

    if (effectTriggered) {
        bc.player.increaseMaxHp(upgraded ? 4 : 3);
    }

    bc.checkCombat();
}

Action Actions::FiendFireAction(int targetIdx, int calculatedDamage) {
    return {queued::FiendFireAction{targetIdx, calculatedDamage}};
}

void queued::FiendFireAction::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.addToTop(Actions::AttackEnemy(targetIdx, calculatedDamage));
    }

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.addToTop( Actions::ExhaustRandomCardInHand(1) );
    }
}

Action Actions::HandOfGreedAction(int idx, int damage, bool upgraded) {
    return {queued::HandOfGreedAction{idx, damage, upgraded}};
}

void queued::HandOfGreedAction::execute(BattleContext& bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].damage(bc, damage);

    const bool effectTriggered = !m.hasStatus<MS::MINION>()
            && !m.isAlive()
            && !m.isHalfDead()
            && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

    if (effectTriggered) {
        bc.player.gainGold(bc, upgraded ? 25 : 20);
    }

    bc.checkCombat();
}

Action Actions::LimitBreakAction() {
    return {queued::LimitBreakAction{}};
}

void queued::LimitBreakAction::execute(BattleContext& bc) const {
    bc.player.buff<PS::STRENGTH>(bc.player.getStatus<PS::STRENGTH>());
}

Action Actions::ReaperAction(int baseDamage) {
    return {queued::ReaperAction{baseDamage}};
}

void queued::ReaperAction::execute(BattleContext& bc) const {
    int healAmount = 0;
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        auto &m = bc.monsters.arr[i];
        if (m.isDeadOrEscaped()) {
            continue;
        }
        int preDamageHp = m.curHp;
        m.attacked(bc, bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage));
        healAmount += preDamageHp-m.curHp;
    }

    if (healAmount > 0) {
        bc.addToBot( Actions::HealPlayer(healAmount) );
    }

    //if (AbstractDungeon.getCurrRoom().monsters.areMonstersBasicallyDead()) {
    //                AbstractDungeon.actionManager.clearPostCombatActions();
    //            }
}

Action Actions::RitualDaggerAction(int idx, int damage) {
    return {queued::RitualDaggerAction{idx, damage}};
}

void queued::RitualDaggerAction::execute(BattleContext& bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].attacked(bc, damage);

    const bool shouldUpgrade = !m.hasStatus<MS::MINION>()
                               && !m.isAlive()
                               && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);
    if (shouldUpgrade) {
        auto &c = bc.curCardQueueItem.card;
        const auto upgradeAmt = c.isUpgraded() ? 5 : 3;

        if (bc.curCardQueueItem.purgeOnUse) {
            bc.cards.findAndUpgradeSpecialData(c.uniqueId, upgradeAmt);
        }
        c.specialData += upgradeAmt;
    }

    bc.checkCombat();
}

Action Actions::SecondWindAction(int blockPerCard) {
    return {queued::SecondWindAction{blockPerCard}};
}

void queued::SecondWindAction::execute(BattleContext& bc) const {
    int cardIdxsToExhaust[10];
    int toExhaustCount = 0;

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &c = bc.cards.hand[i];
        if (c.getType() != CardType::ATTACK) {
            cardIdxsToExhaust[toExhaustCount++] = i;
            bc.addToTop( Actions::GainBlock(blockPerCard) );
        }
    }

    for (int i = 0; i < toExhaustCount; ++i) {
        const auto handIdx = cardIdxsToExhaust[i];
        const auto &c = bc.cards.hand[handIdx];

        bc.addToTop( Actions::ExhaustSpecificCardInHand(handIdx, c.uniqueId) );
    }
}

Action Actions::SeverSoulExhaustAction() {
    return {queued::SeverSoulExhaustAction{}};
}

void queued::SeverSoulExhaustAction::execute(BattleContext& bc) const {
    for (int i = bc.cards.cardsInHand-1; i >= 0; --i) {
        const auto &c = bc.cards.hand[i];
        if (c.getType() != CardType::ATTACK) {
            bc.addToBot( Actions::ExhaustSpecificCardInHand(i, c.getUniqueId()) );
        }
    }
}

Action Actions::SwordBoomerangAction(int baseDamage) {
    return {queued::SwordBoomerangAction{baseDamage}};
}

void queued::SwordBoomerangAction::execute(BattleContext& bc) const {
    const static CardInstance swordBoomerang {CardId::SWORD_BOOMERANG};
    const auto idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }

    int damage = bc.calculateCardDamage(swordBoomerang, idx, baseDamage);
    bc.addToTop(Actions::AttackEnemy(idx, damage));
}

Action Actions::SpotWeaknessAction(int target, int strength) {
    return {queued::SpotWeaknessAction{target, strength}};
}

void queued::SpotWeaknessAction::execute(BattleContext& bc) const {
    bc.noteEnemyIntentAccess();
    if (bc.monsters.arr[target].isAttacking()) {
        bc.player.buff<PS::STRENGTH>(strength);
    }
}

Action Actions::ScryAction(int count) {
    return {queued::ScryAction{count}};
}

void queued::ScryAction::execute(BattleContext& bc) const {
    const int visible = std::min(count, static_cast<int>(bc.cards.drawPile.size()));
    if (visible == 0) return;
    bc.noteDrawPileOrderAccess();
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::SCRY;
    bc.cardSelectInfo.pickCount = visible;
    bc.inputState = InputState::CARD_SELECT;
}

Action Actions::BouncingFlaskAction(int poison) {
    return {queued::BouncingFlaskAction{poison}};
}

void queued::BouncingFlaskAction::execute(BattleContext& bc) const {
    for (int i = 0; i < 3; ++i) {
        fixed_list<int, 5> targets;
        for (int idx = 0; idx < bc.monsters.monsterCount; ++idx) {
            if (!bc.monsters.arr[idx].isDeadOrEscaped()) {
                targets.push_back(idx);
            }
        }
        if (targets.empty()) {
            return;
        }
        const int target = targets[bc.cardRandomRng.random(targets.size() - 1)];
        bc.monsters.arr[target].buff<MS::POISON>(poison);
    }
}

Action Actions::CalculatedGambleAction() {
    return {queued::CalculatedGambleAction{}};
}

void queued::CalculatedGambleAction::execute(BattleContext& bc) const {
    const int count = bc.cards.cardsInHand;
    for (int i = count - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.moveToDiscardPile(card);
        bc.onManualDiscard(card);
    }
    bc.addToBot(Actions::DrawCards(count));
}

Action Actions::EscapePlanAction(int block) {
    return {queued::EscapePlanAction{block}};
}

void queued::EscapePlanAction::execute(BattleContext& bc) const {
    if (!bc.cards.drawPile.empty()) {
        bc.noteDrawPileOrderAccess();
        const auto &card = bc.cards.drawPile.back();
        if (card.getType() == CardType::ATTACK) {
            bc.addToBot(Actions::GainBlock(bc.calculateCardBlock(block)));
        }
    }
    bc.addToBot(Actions::DrawCards(1));
}

Action Actions::RandomDiscardAction() {
    return {queued::RandomDiscardAction{}};
}

void queued::RandomDiscardAction::execute(BattleContext& bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    const int idx = bc.cardRandomRng.random(bc.cards.cardsInHand - 1);
    const auto card = bc.cards.hand[idx];
    bc.cards.removeFromHandAtIdx(idx);
    bc.cards.moveToDiscardPile(card);
    bc.onManualDiscard(card);
}

Action Actions::SetupAction() {
    return {queued::SetupAction{}};
}

void queued::SetupAction::execute(BattleContext& bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    bc.openSimpleCardSelectScreen(CardSelectTask::SETUP, 1);
}

Action Actions::SunderAction(int target, int damage) {
    return {queued::SunderAction{target, damage}};
}

void queued::SunderAction::execute(BattleContext& bc) const {
    auto &monster = bc.monsters.arr[target];
    if (monster.isDeadOrEscaped()) {
        return;
    }

    monster.attacked(bc, damage);
    if (!monster.isAlive() && !monster.isHalfDead()) {
        bc.player.gainEnergy(3);
    }
    bc.checkCombat();
}

Action Actions::ModifyCurrentCardCost(int amount) {
    return {queued::ModifyCurrentCardCost{amount}};
}

void queued::ModifyCurrentCardCost::execute(BattleContext& bc) const {
    bc.curCardQueueItem.card.updateCost(amount);
}

Action Actions::StormOfSteelAction(bool upgraded) {
    return {queued::StormOfSteelAction{upgraded}};
}

void queued::StormOfSteelAction::execute(BattleContext& bc) const {
    const int discarded = bc.cards.cardsInHand;
    for (int i = bc.cards.cardsInHand - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.moveToDiscardPile(card);
        bc.onManualDiscard(card);
    }
    bc.addToBot(Actions::MakeTempCardInHand(CardId::SHIV, upgraded, discarded));
}

Action Actions::BulletTimeAction() {
    return {queued::BulletTimeAction{}};
}

void queued::BulletTimeAction::execute(BattleContext& bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.cards.hand[i].setCostForTurn(0);
    }
    bc.player.buff<PS::NO_DRAW>(1);
}

Action Actions::WhirlwindAction(int baseDamage, int energy, bool useEnergy) {
    return {queued::WhirlwindAction{baseDamage, energy, useEnergy}};
}

void queued::WhirlwindAction::execute(BattleContext& bc) const {
    // assume bc.curCard is the card being used

    if (useEnergy) {
        bc.player.useEnergy(bc.player.energy);
    }

    DamageMatrix damageMatrix {0};
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            const auto calcDamage = bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage);

            damageMatrix[i] = static_cast<std::uint16_t>( // fit damage into uint16
                std::min(
                    static_cast<int>(std::numeric_limits<std::uint16_t>::max()),
                    calcDamage
                )
            );
        }
    }

    const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);
    if (effectAmount > 0) {
        Actions::AttackAllMonsterRecursive(damageMatrix, effectAmount).execute(bc);
    }
}

Action Actions::AttackAllMonsterRecursive(DamageMatrix matrix, int timesRemaining) {
    return {queued::AttackAllMonsterRecursive{matrix, timesRemaining}};
}

void queued::AttackAllMonsterRecursive::execute(BattleContext& bc) const {
    if (timesRemaining <= 0) {
        return;
    }

    Actions::AttackAllEnemy(matrix).execute(bc);

    if (timesRemaining > 1) {
        bc.addToTop(Actions::AttackAllMonsterRecursive(matrix, timesRemaining-1)); // todo should this be to the top? test with
    }
}

Action Actions::Judgment(int target, int threshold) {
    return {queued::Judgment{target, threshold}};
}

void queued::Judgment::execute(BattleContext& bc) const {
    auto& monster = bc.monsters.arr[target];
    if (!monster.isDeadOrEscaped() && monster.curHp <= threshold) {
        monster.curHp = 0;
        monster.die(bc);
        bc.checkCombat();
    }
}

Action Actions::Catalyst(int target, int multiplier) {
    return {queued::Catalyst{target, multiplier}};
}

void queued::Catalyst::execute(BattleContext& bc) const {
    auto& monster = bc.monsters.arr[target];
    if (!monster.isDeadOrEscaped() && monster.hasStatus<MS::POISON>()) {
        monster.setStatus<MS::POISON>(monster.getStatus<MS::POISON>() * multiplier);
    }
}

void queued::BuffPlayer::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        if (s == PlayerStatus::CORRUPTION && !bc.player.hasStatus<PS::CORRUPTION>()) {
            bc.cards.onBuffCorruption();
        }
        bc.player.buff<s>(amount);
    }, std::make_index_sequence<std::size(playerStatusStrings)>{});
}

void queued::DebuffPlayer::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        bc.player.debuff<s>(amount, isSourceMonster);
    }, std::make_index_sequence<std::size(playerStatusStrings)>{});
}

void queued::DecrementStatus::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        bc.player.decrementStatus<s>(amount);
    }, std::make_index_sequence<std::size(playerStatusStrings)>{});
}

void queued::RemoveStatus::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        bc.player.setHasStatus<s>(false);
    }, std::make_index_sequence<std::size(playerStatusStrings)>{});
}

void queued::BuffEnemy::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        // todo check if alive?
        bc.monsters.arr[idx].buff<s>(amount);
    }, std::make_index_sequence<std::size(monsterStatusEnumStrings)>{});
}

void queued::DebuffEnemy::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        bc.debuffEnemy<s>(idx, amount, isSourceMonster);
    }, std::make_index_sequence<std::size(monsterStatusEnumStrings)>{});
}

void queued::DebuffAllEnemy::execute(BattleContext& bc) const {
    visit_status(status, [&](auto tag) {
        constexpr auto s = decltype(tag)::value;
        for (int i = bc.monsters.monsterCount-1; i >= 0; --i) {
            if (bc.monsters.arr[i].isTargetable()) {
                bc.addToTop(Actions::DebuffEnemy<s>(i, amount, isSourceMonster));
            }
        }
    }, std::make_index_sequence<std::size(monsterStatusEnumStrings)>{});
}

void Action::execute(BattleContext& state) const {
    std::visit([&](const auto& effect) { effect.execute(state); }, payload);
}

} // namespace sts
