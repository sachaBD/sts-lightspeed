//
// Created by gamerpuppy on 7/4/2021.
//

#include "combat/Actions.h"
#include "combat/BattleContext.h"
#include "game/Game.h"

namespace sts {

void Actions::SetState(BattleContext& bc, InputState state) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.setState(state);
}

void Actions::AttackEnemy(BattleContext& bc, int idx, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].attacked(bc, damage);
    if (bc.player.hasStatus<PS::ENVENOM>()) {
        bc.monsters.arr[idx].buff<MS::POISON>(bc.player.getStatus<PS::ENVENOM>());
    }
    bc.checkCombat();
}

void Actions::AttackAllEnemy(BattleContext& bc, int baseDamage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    // assume bc.curCard is the card being used

    int damageMatrix[5];
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            damageMatrix[i] = bc.calculateCardDamage(bc.currentCard.card, i, baseDamage);
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

void Actions::AttackAllEnemy(BattleContext& bc, DamageMatrix damageMatrix) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::DamageEnemy(BattleContext& bc, int idx, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

void Actions::DamageAllEnemy(BattleContext& bc, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int idx = 0; idx < bc.monsters.monsterCount; idx++) {
        if (!bc.monsters.arr[idx].isDeadOrEscaped()) {
            bc.monsters.arr[idx].damage(bc, damage);
        }
    }
    bc.checkCombat();
}

void Actions::AttackPlayer(BattleContext& bc, int idx, int damage) {
    bc.player.attacked(bc, idx, damage);
}

void Actions::DamagePlayer(BattleContext& bc, int damage, bool selfDamage) {
    bc.player.damage(bc, damage, selfDamage);
}

void Actions::VampireAttack(BattleContext& bc, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const auto mIdx = 0;
    auto &m = bc.monsters.arr[mIdx]; // only used by shelled parasite so idx is 0
    bc.player.attacked(bc, mIdx, damage);
    if (m.isAlive()) {
        m.heal(std::min(damage, static_cast<int>(bc.player.lastAttackUnblockedDamage)));
    }
}

void Actions::PlayerLoseHp(BattleContext& bc, int hp, bool selfDamage) {
    bc.player.loseHp(bc, hp, selfDamage);
}

void Actions::HealPlayer(BattleContext& bc, int amount) {
    bc.player.heal(amount);
}

void Actions::MonsterGainBlock(BattleContext& bc, int idx, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.monsters.arr[idx].block += amount;
}

void Actions::RollMove(BattleContext& bc, int monsterIdx) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    Monster &m = bc.monsters.arr[monsterIdx];
    m.rollMove(bc);
}

void Actions::ReactiveRollMove(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    // writhing mass is always monster 0
    Monster &m = bc.monsters.arr[0];

    for (int i = 0 ; i < m.getStatus<MS::REACTIVE>(); ++i) {
        m.rollMove(bc);
    }
    m.setStatus<MS::REACTIVE>(0);
}

void Actions::NoOpRollMove(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.noOpRollMove();
}

void Actions::ChangeStance(BattleContext& bc, Stance stance) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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
        Actions::GainBlock(bc, bc.player.nirvanaBlock);
    }
    if (bc.player.mentalFortressBlock > 0) {
        Actions::GainBlock(bc, bc.player.mentalFortressBlock);
    }
    bc.player.stance = stance;
    if (stance == Stance::WRATH && bc.player.rushdownDraw > 0) {
        Actions::DrawCards(bc, bc.player.rushdownDraw);
    }
    if (stance == Stance::DIVINITY) {
        bc.player.gainEnergy(3);
    }
}

void Actions::GainEnergy(BattleContext& bc, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.gainEnergy(amount);
}

void Actions::GainBlock(BattleContext& bc, int amount) {
    bc.player.gainBlock(bc, amount);
}
void Actions::DrawCards(BattleContext& bc, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.drawCards(amount);
}

void Actions::EmptyDeckShuffle(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    java::Collections::shuffle(
            bc.cards.discardPile.begin(),
            bc.cards.discardPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );

    bc.cards.moveDiscardPileIntoToDrawPile();
}

void Actions::ShuffleDrawPile(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    java::Collections::shuffle(
            bc.cards.drawPile.begin(),
            bc.cards.drawPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );
}

void Actions::ShuffleTempCardIntoDrawPile(BattleContext& bc, CardId id, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    CardInstance c(id);
    for (int i = 0; i < count; ++i) {
        const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
        bc.cards.createTempCardInDrawPile(idx, c);
    }
}

void Actions::PlayTopCard(BattleContext& bc, int monsterTargetIdx, bool exhausts) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.playTopCardInDrawPile(monsterTargetIdx, exhausts);
}


// todo fix the arguments are used wrongly all over ? what did this mean
// todo check for master reality
void Actions::MakeTempCardInHand(BattleContext& bc, CardId card, bool upgraded, int amount) {
    CardInstance c(card, upgraded);
    return Actions::MakeTempCardInHand(bc, c, amount);
}

void Actions::MakeTempCardInHand(BattleContext& bc, CardInstance card, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < amount; ++i) {
        CardInstance c(card);
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}

void Actions::MakeTempCardInDrawPile(BattleContext& bc, const CardInstance &c, int amount, bool shuffleInto) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < amount; ++i) {
        if (shuffleInto) {
            const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
            bc.cards.createTempCardInDrawPile(idx, c);
        }
        // todo else
    }
}

void Actions::MakeTempCardInDiscard(BattleContext& bc, const CardInstance &c, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
           for (int i = 0; i < amount; ++i) {
               bc.cards.createTempCardInDiscard(c);
           }
}

void Actions::MakeTempCardsInHand(BattleContext& bc, std::vector<CardInstance> cards) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (auto c : cards) {
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}


void Actions::DiscardNoTriggerCard(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const auto &c = bc.currentCard.card;
    bc.cards.notifyRemoveFromHand(c);
    bc.cards.moveToDiscardPile(c);
}


void Actions::TriggerEndOfTurnOrbsAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.triggerEndOfTurnOrbs(bc);
}

void Actions::ChannelOrb(BattleContext& bc, Orb orb, int orbData) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.channelOrb(bc, orb, orbData);
}

void Actions::EvokeFrontOrb(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.evokeFrontOrb(bc);
}

void Actions::ExhaustTopCardInHand(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.exhaustTopCardInHand();
}

void Actions::ExhaustSpecificCardInHand(BattleContext& bc, int idx, std::int16_t uniqueId) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.exhaustSpecificCardInHand(idx, uniqueId);
}

void Actions::DamageRandomEnemy(BattleContext& bc, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const int idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }
    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

void Actions::ExhaustRandomCardInHand(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::MadnessAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::RandomizeHandCost(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.cost >= 0) {
            int newCost = bc.cardRandomRng.random(3);
            c.cost = newCost;
            c.costForTurn = newCost;
        }
    }
}

void Actions::GainBlockRandomEnemy(BattleContext& bc, int sourceMonster, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::SummonGremlins(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::SpawnTorchHeads(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::SpireShieldDebuff(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.aiRng.randomBoolean()) {
        bc.player.debuff<PS::FOCUS>(-1);
    } else {
        bc.player.debuff<PS::STRENGTH>(-1);
    }
}


void Actions::PutRandomCardsInDrawPile(BattleContext& bc, CardType type, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::DiscoveryAction(BattleContext& bc, CardType type, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.haveUsedDiscoveryAction = true;
    bc.openDiscoveryScreen(sts::generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, type), amount);
}

void Actions::InfernalBladeAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const auto cid = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, CardType::ATTACK);
    CardInstance c(cid);
    c.setCostForTurn(0);
    Actions::MakeTempCardInHand(bc, c);
}

void Actions::JackOfAllTradesAction(BattleContext& bc, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const auto c1 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
    Actions::MakeTempCardInHand(bc, c1);
    if (upgraded) {
        auto c2 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
        Actions::MakeTempCardInHand(bc, c2);
    }
}

void Actions::TransmutationAction(BattleContext& bc, bool upgraded, int energy, bool useEnergy) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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
    Actions::MakeTempCardsInHand(bc, cards);

    if (useEnergy) {
        bc.player.useEnergy(bc.player.energy);
    }
}

void Actions::ViolenceAction(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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
void Actions::BetterDiscardPileToHandAction(BattleContext& bc, int amount, CardSelectTask task) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::ArmamentsAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::DualWieldAction(BattleContext& bc, int copyCount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::ExhumeAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::ForethoughtAction(BattleContext& bc, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::HeadbuttAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.cards.discardPile.empty()) {
        return;

    } else if (bc.cards.discardPile.size() == 1) {
        bc.chooseHeadbuttCard(0);
    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::HEADBUTT, 1);
    }
}

void Actions::ChooseExhaustOne(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.cards.cardsInHand == 0) {
        return;

    } else if (bc.cards.cardsInHand == 1) {
        bc.chooseExhaustOneCard(0);

    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::EXHAUST_ONE, 1);

    }
}

void Actions::DrawToHandAction(BattleContext& bc, CardSelectTask task, CardType cardType) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::WarcryAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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


void Actions::UpgradeAllCardsInHand(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {  // curses and statuses can't be upgraded
            bc.cards.hand[i].upgrade();
        }
    }
}

void Actions::EssenceOfDarkness(BattleContext& bc, int darkOrbsPerSlot) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const auto count = bc.player.orbSlots * darkOrbsPerSlot;
    for (int i = 0; i < count; ++i) {
        bc.player.channelOrb(bc, Orb::DARK);
    }
}

void Actions::MakeRandomCardInHand(BattleContext& bc, CardType type, int count, bool zeroCost) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < count; ++i) {
        CardInstance card(getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, type));
        if (zeroCost) {
            card.setCostForTurn(0);
        }
        bc.moveToHandHelper(card);
    }
}

void Actions::MakeRandomCommonCardInHand(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < count; ++i) {
        CardId id = CardId::INVALID;
        do {
            id = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc);
        } while (getCardRarity(id) != CardRarity::COMMON);
        bc.moveToHandHelper(CardInstance(id));
    }
}

void Actions::DiscardOne(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::RecycleAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    if (bc.cards.cardsInHand == 1) {
        bc.chooseRecycleCard(0);
        return;
    }
    bc.openSimpleCardSelectScreen(CardSelectTask::RECYCLE, 1);
}

void Actions::ScrapeAction(BattleContext& bc, int drawCount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::RebootAction(BattleContext& bc, int drawCount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = bc.cards.cardsInHand - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.shuffleIntoDrawPile(bc.cardRandomRng, card);
    }
    bc.drawCards(drawCount);
}

void Actions::SeekAction(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::AllForOneAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::IncreaseOrbSlots(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.increaseOrbSlots(count);
}

void Actions::SuicideAction(BattleContext& bc, int monsterIdx, bool triggerRelics) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    auto &m = bc.monsters.arr[monsterIdx];
    if (triggerRelics) {
        if (m.isAlive()) {
            m.damage(bc, m.curHp);
        }
    } else {
        bc.monsters.arr[monsterIdx].suicideAction(bc);
    }
}

void Actions::PoisonLoseHpAction(BattleContext& bc, int monsterIdx) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::RemovePlayerDebuffs(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.removeDebuffs();
}

void Actions::UpgradeRandomCardAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::CodexAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::CODEX;
    bc.cardSelectInfo.codexCards() =
            generateDiscoveryCards(bc.cardRandomRng, CharacterClass::IRONCLAD, CardType::INVALID);
}

void Actions::ExhaustMany(BattleContext& bc, int limit) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::EXHAUST_MANY;
    bc.cardSelectInfo.pickCount = limit;
}

void Actions::GambleAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::GAMBLE;
}

void Actions::ToolboxAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::DISCOVERY;
    bc.cardSelectInfo.discovery_CopyCount() = 1;
    bc.cardSelectInfo.discovery_Cards() =
            generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, CardType::STATUS); // status is mapped to colorless
}

void Actions::DualityAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.buff<PS::DEXTERITY>(1);
    bc.player.debuff<PS::LOSE_DEXTERITY>(1);
}

void Actions::ApotheosisAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::DropkickAction(BattleContext& bc, int targetIdx) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.monsters.arr[targetIdx].isTargetable() && bc.monsters.arr[targetIdx].hasStatus<MS::VULNERABLE>()) {
        Actions::DrawCards(bc, 1);
        Actions::GainEnergy(bc, 1);
    }

    const auto &c = bc.currentCard.card;
    const int damage = bc.calculateCardDamage(c, targetIdx, c.isUpgraded() ? 8 : 5);
    Actions::AttackEnemy(bc, targetIdx, damage);
}

void Actions::EnlightenmentAction(BattleContext& bc, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::EntrenchAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.gainBlock(bc, bc.player.block);
}

void Actions::FeedAction(BattleContext& bc, int idx, int damage, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::FiendFireAction(BattleContext& bc, int targetIdx, int calculatedDamage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const std::vector<CardInstance> original(bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
    for (auto card : original) {
        if (bc.outcome != Outcome::UNDECIDED) break;
        bc.cards.removeFromHandById(card.uniqueId);
        bc.triggerAndMoveToExhaustPile(card);
    }
    for (std::size_t i = 0; i < original.size(); ++i) Actions::AttackEnemy(bc, targetIdx, calculatedDamage);
}

void Actions::HandOfGreedAction(BattleContext& bc, int idx, int damage, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::LimitBreakAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.player.buff<PS::STRENGTH>(bc.player.getStatus<PS::STRENGTH>());
}

void Actions::ReaperAction(BattleContext& bc, int baseDamage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    int healAmount = 0;
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        auto &m = bc.monsters.arr[i];
        if (m.isDeadOrEscaped()) {
            continue;
        }
        int preDamageHp = m.curHp;
        m.attacked(bc, bc.calculateCardDamage(bc.currentCard.card, i, baseDamage));
        healAmount += preDamageHp-m.curHp;
    }

    if (healAmount > 0) {
        Actions::HealPlayer(bc, healAmount);
    }

    //if (AbstractDungeon.getCurrRoom().monsters.areMonstersBasicallyDead()) {
    //                AbstractDungeon.actionManager.clearPostCombatActions();
    //            }
}

void Actions::RitualDaggerAction(BattleContext& bc, int idx, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].attacked(bc, damage);

    const bool shouldUpgrade = !m.hasStatus<MS::MINION>()
                               && !m.isAlive()
                               && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);
    if (shouldUpgrade) {
        auto &c = bc.currentCard.card;
        const auto upgradeAmt = c.isUpgraded() ? 5 : 3;

        if (bc.currentCard.purgeOnUse) {
            bc.cards.findAndUpgradeSpecialData(c.uniqueId, upgradeAmt);
        }
        c.specialData += upgradeAmt;
    }

    bc.checkCombat();
}

void Actions::SecondWindAction(BattleContext& bc, int blockPerCard) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const std::vector<CardInstance> original(bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
    for (auto card : original) {
        if (bc.outcome != Outcome::UNDECIDED) break;
        if (card.getType() == CardType::ATTACK) continue;
        bc.cards.removeFromHandById(card.uniqueId);
        bc.triggerAndMoveToExhaustPile(card);
        Actions::GainBlock(bc, blockPerCard);
    }
}

void Actions::SeverSoulExhaustAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = bc.cards.cardsInHand-1; i >= 0; --i) {
        const auto &c = bc.cards.hand[i];
        if (c.getType() != CardType::ATTACK) {
            Actions::ExhaustSpecificCardInHand(bc, i, c.getUniqueId());
        }
    }
}

void Actions::SwordBoomerangAction(BattleContext& bc, int baseDamage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const static CardInstance swordBoomerang {CardId::SWORD_BOOMERANG};
    const auto idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }

    int damage = bc.calculateCardDamage(swordBoomerang, idx, baseDamage);
    Actions::AttackEnemy(bc, idx, damage);
}

void Actions::SpotWeaknessAction(BattleContext& bc, int target, int strength) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.noteEnemyIntentAccess();
    if (bc.monsters.arr[target].isAttacking()) {
        bc.player.buff<PS::STRENGTH>(strength);
    }
}

void Actions::ScryAction(BattleContext& bc, int count) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const int visible = std::min(count, static_cast<int>(bc.cards.drawPile.size()));
    if (visible == 0) return;
    bc.noteDrawPileOrderAccess();
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::SCRY;
    bc.cardSelectInfo.pickCount = visible;
    bc.inputState = InputState::CARD_SELECT;
}

void Actions::BouncingFlaskAction(BattleContext& bc, int poison) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::CalculatedGambleAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const int count = bc.cards.cardsInHand;
    for (int i = count - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.moveToDiscardPile(card);
        bc.onManualDiscard(card);
    }
    Actions::DrawCards(bc, count);
}

void Actions::EscapePlanAction(BattleContext& bc, int block) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (!bc.cards.drawPile.empty()) {
        bc.noteDrawPileOrderAccess();
        const auto &card = bc.cards.drawPile.back();
        if (card.getType() == CardType::ATTACK) {
            Actions::GainBlock(bc, bc.calculateCardBlock(block));
        }
    }
    Actions::DrawCards(bc, 1);
}

void Actions::RandomDiscardAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    const int idx = bc.cardRandomRng.random(bc.cards.cardsInHand - 1);
    const auto card = bc.cards.hand[idx];
    bc.cards.removeFromHandAtIdx(idx);
    bc.cards.moveToDiscardPile(card);
    bc.onManualDiscard(card);
}

void Actions::SetupAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (bc.cards.cardsInHand == 0) {
        return;
    }
    bc.openSimpleCardSelectScreen(CardSelectTask::SETUP, 1);
}

void Actions::SunderAction(BattleContext& bc, int target, int damage) {
    if (bc.outcome != Outcome::UNDECIDED) return;
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

void Actions::ModifyCurrentCardCost(BattleContext& bc, int amount) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    bc.currentCard.card.updateCost(amount);
}

void Actions::StormOfSteelAction(BattleContext& bc, bool upgraded) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    const int discarded = bc.cards.cardsInHand;
    for (int i = bc.cards.cardsInHand - 1; i >= 0; --i) {
        const auto card = bc.cards.hand[i];
        bc.cards.removeFromHandAtIdx(i);
        bc.cards.moveToDiscardPile(card);
        bc.onManualDiscard(card);
    }
    Actions::MakeTempCardInHand(bc, CardId::SHIV, upgraded, discarded);
}

void Actions::BulletTimeAction(BattleContext& bc) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.cards.hand[i].setCostForTurn(0);
    }
    bc.player.buff<PS::NO_DRAW>(1);
}

void Actions::WhirlwindAction(BattleContext& bc, int baseDamage, int energy, bool useEnergy) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    // assume bc.curCard is the card being used

    if (useEnergy) {
        bc.player.useEnergy(bc.player.energy);
    }

    DamageMatrix damageMatrix {0};
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            const auto calcDamage = bc.calculateCardDamage(bc.currentCard.card, i, baseDamage);

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
        Actions::AttackAllMonsterRecursive(bc, damageMatrix, effectAmount);
    }
}

void Actions::AttackAllMonsterRecursive(BattleContext& bc, DamageMatrix matrix, int timesRemaining) {
    if (bc.outcome != Outcome::UNDECIDED) return;
    if (timesRemaining <= 0) {
        return;
    }

    Actions::AttackAllEnemy(bc, matrix);

    if (timesRemaining > 1) {
        Actions::AttackAllMonsterRecursive(bc, matrix, timesRemaining-1); // todo should this be to the top? test with
    }
}

} // namespace sts
