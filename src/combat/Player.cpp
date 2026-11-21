//
// Created by gamerpuppy on 7/4/2021.
//

#include <cassert>

#include <constants/PlayerStatusEffects.h>
#include <combat/BattleContext.h>
#include <combat/Actions.h>
#include "combat/Player.h"

namespace sts {


bool Player::hasRelicRuntime(RelicId r) const {
    if ((int) r < 64) {
        return relicBits0 & (1ULL << (int)r);
    } else {
        return relicBits1 & (1ULL << ((int)r-64));
    }
}

bool Player::hasStatusRuntime(PlayerStatus s) const {
    switch (s) {
        case PS::ARTIFACT:
            return artifact;

        case PS::DEXTERITY:
            return dexterity;

        case PS::FOCUS:
            return focus;

        case PS::STRENGTH:
            return strength;

        default:
            break;
    }

    int idx = static_cast<int>(s);
    if (idx < 64) {
        return statusBits0 & (1ULL << idx);
    } else {
        return statusBits1 & (1ULL << (idx-64));
    }
}

int Player::getStatusRuntime(PlayerStatus s) const {
    switch (s) {
        case PS::ARTIFACT:
            return artifact;
        case PS::DEXTERITY:
            return dexterity;
        case PS::FOCUS:
            return focus;
        case PS::STRENGTH:
            return strength;
        default:
            if (hasStatusRuntime(s)) {
                return statusMap.at(s);
            } else {
                return 0;
            }
    }
}

void Player::gainBlock(BattleContext &bc, int amount) {
    if (amount <= 0) {
        return;
    }

    // AbstractCreature.addBlock caps block at 999. Bound the increment before
    // addition as well: repeated Entrench must not overflow the simulator int.
    block = std::min(999, block + std::min(amount, 999));

    if (hasStatus<PS::JUGGERNAUT>()) {
        Actions::DamageRandomEnemy(bc, getStatus<PS::JUGGERNAUT>());
    }

    // todo watcher weak power
}

void Player::gainGold(BattleContext &bc, int amount) {
#ifdef sts_asserts
    assert(amount > 0);
#endif

    if (hasRelic<R::ECTOPLASM>()) {
        return;
    }

    gold += amount;
    if (hasRelic<R::BLOODY_IDOL>()) {
        heal(5);
    }
}

void Player::useEnergy(int amount) {
#ifdef sts_asserts
    assert(energy >= amount);
#endif

    energy -= amount;
}

void Player::gainEnergy(int amount) {
    energy += amount;
}

void Player::increaseOrbSlots(int amount) {
    const auto oldSlots = orbSlots;
    orbSlots = static_cast<int8_t>(std::min<int>(static_cast<int>(orbs.size()), orbSlots + amount));
    for (int i = oldSlots; i < orbSlots; ++i) {
        orbs[i] = Orb::EMPTY;
        orbData[i] = 0;
    }
}

void Player::channelOrb(BattleContext &bc, Orb orb, int orbDataValue) {
    if (orb == Orb::EMPTY) {
        return;
    }
    if (orb == Orb::LIGHTNING) {
        ++lightningOrbsChanneled;
    } else if (orb == Orb::FROST) {
        ++frostOrbsChanneled;
    }
    if (orbSlots <= 0) {
        return;
    }
    for (int i = 0; i < orbSlots; ++i) {
        if (orbs[i] == Orb::EMPTY) {
            orbs[i] = orb;
            orbData[i] = orb == Orb::DARK ? (orbDataValue > 0 ? orbDataValue : 6 + focus) : (orb == Orb::FUSION ? std::max(1, orbDataValue) : 0);
            return;
        }
    }

    evokeFrontOrb(bc);
    orbs[orbSlots - 1] = orb;
    orbData[orbSlots - 1] = orb == Orb::DARK ? (orbDataValue > 0 ? orbDataValue : 6 + focus) : (orb == Orb::FUSION ? std::max(1, orbDataValue) : 0);
}

bool Player::hasEmptyOrb() const {
    for (int i = 0; i < orbSlots; ++i) {
        if (orbs[i] == Orb::EMPTY) {
            return true;
        }
    }
    return false;
}

void Player::evokeFrontOrb(BattleContext &bc) {
    if (orbSlots <= 0 || orbs[0] == Orb::EMPTY) {
        return;
    }
    const auto orb = orbs[0];
    const auto damage = orb == Orb::DARK ? orbData[0] : 8 + focus;
    if (orb == Orb::LIGHTNING || orb == Orb::DARK) {
        if (orb == Orb::LIGHTNING && hasStatus<PS::ELECTRO>()) {
            for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                if (!bc.monsters.arr[i].isDeadOrEscaped()) {
                    const int orbDamage = bc.monsters.arr[i].hasStatus<MS::LOCK_ON>() ? damage * 3 / 2 : damage;
                    bc.monsters.arr[i].damage(bc, orbDamage);
                }
            }
            bc.checkCombat();
        } else {
            const auto target = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng);
            if (target >= 0) {
                const int orbDamage = bc.monsters.arr[target].hasStatus<MS::LOCK_ON>() ? damage * 3 / 2 : damage;
                bc.monsters.arr[target].damage(bc, orbDamage);
                bc.checkCombat();
            }
        }
    } else if (orb == Orb::FROST) {
        gainBlock(bc, 5 + focus);
    } else if (orb == Orb::FUSION) {
        gainEnergy(2);
    }
    for (int i = 1; i < orbSlots; ++i) {
        orbs[i - 1] = orbs[i];
        orbData[i - 1] = orbData[i];
    }
    orbs[orbSlots - 1] = Orb::EMPTY;
    orbData[orbSlots - 1] = 0;
}

void Player::triggerOrbEndOfTurn(BattleContext &bc, int slot) {
    if (slot < 0 || slot >= orbSlots) {
        return;
    }
    switch (orbs[slot]) {
        case Orb::LIGHTNING: {
            const int damage = 3 + focus;
            if (hasStatus<PS::ELECTRO>()) {
                for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                    if (!bc.monsters.arr[i].isDeadOrEscaped()) {
                        const int orbDamage = bc.monsters.arr[i].hasStatus<MS::LOCK_ON>() ? damage * 3 / 2 : damage;
                        bc.monsters.arr[i].damage(bc, orbDamage);
                    }
                }
                bc.checkCombat();
            } else {
                const auto target = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng);
                if (target >= 0) {
                    const int orbDamage = bc.monsters.arr[target].hasStatus<MS::LOCK_ON>() ? damage * 3 / 2 : damage;
                    bc.monsters.arr[target].damage(bc, orbDamage);
                    bc.checkCombat();
                }
            }
            break;
        }
        case Orb::FROST:
            gainBlock(bc, 2 + focus);
            break;
        case Orb::DARK:
            orbData[slot] += 6 + focus;
            break;
        case Orb::EMPTY:
        case Orb::FUSION:
            break;
    }
}

void Player::triggerOrbStartOfTurn(BattleContext &bc, int slot) {
    if (slot >= 0 && slot < orbSlots && orbs[slot] == Orb::FUSION) {
        gainEnergy(orbData[slot]);
    }
}

void Player::triggerEndOfTurnOrbs(BattleContext &bc) {
    for (int i = 0; i < orbSlots; ++i) {
        triggerOrbEndOfTurn(bc, i);
    }
}

void Player::triggerStartOfTurnOrbs(BattleContext &bc) {
    for (int i = 0; i < orbSlots; ++i) {
        triggerOrbStartOfTurn(bc, i);
    }
}

void Player::removeDebuffs() {
    if (getStatus<PS::STRENGTH>() < 0) {
        setStatusValueNoChecks<PS::STRENGTH>(0);
    }
    if (getStatus<PS::DEXTERITY>() < 0) {
        setStatusValueNoChecks<PS::DEXTERITY>(0);
    }

    removeStatus<PS::BIAS>();
    removeStatus<PS::CONFUSED>();
    removeStatus<PS::CONSTRICTED>();

    if (hasStatus<PS::DRAW_REDUCTION>()) {
        ++cardDrawPerTurn;
        removeStatus<PS::DRAW_REDUCTION>();
    }

    removeStatus<PS::ENTANGLED>();
    removeStatus<PS::FASTING>();
    removeStatus<PS::FRAIL>();
    removeStatus<PS::HEX>();
    removeStatus<PS::LOSE_DEXTERITY>();
    removeStatus<PS::LOSE_STRENGTH>();
    removeStatus<PS::NO_BLOCK>();
    removeStatus<PS::NO_DRAW>();
    removeStatus<PS::VULNERABLE>();
    removeStatus<PS::WEAK>();
    removeStatus<PS::WRAITH_FORM>();
}

void Player::increaseMaxHp(int amount) {
    maxHp += amount;
    heal(amount);
}

void Player::heal(int amount) {
    if (hasRelic<RelicId::MARK_OF_THE_BLOOM>()) {
        return;
    }

    if (hasRelic<RelicId::MAGIC_FLOWER>()) {
        amount = amount * 3 / 2;
    }

    bool wasBloodied = curHp <= maxHp/2;

    curHp = std::min(static_cast<int>(maxHp), curHp + amount);

    if (wasBloodied && curHp > maxHp/2 && hasRelic<RelicId::RED_SKULL>()) {
        debuff<PS::STRENGTH>(3);
    }
}

void Player::damage(BattleContext &bc, const int calculatedDamage, const bool selfDamage) {
#ifdef sts_asserts
    assert(calculatedDamage >= 0);
#endif

    int damage = calculatedDamage;

    if (damage > 0 && hasStatus<PS::INTANGIBLE>()) {
        damage = 1;
    }

    const auto savedBlock = block;
    block = std::max(0, block-damage);
    damage -= savedBlock;



    damage -= block;
    if (damage > 0) {
        block = 0;
    }

    if (damage > 0 && hasStatus<PS::BUFFER>()) {
        decrementStatus<PS::BUFFER>();
        damage = 0;
    }

    if (damage > 0 && hasRelic<RelicId::TUNGSTEN_ROD>()) {
        damage -= 1;
    }

    if (damage > 0) {
        hpWasLost(bc, damage, selfDamage);
    }
}

void Player::attacked(BattleContext &bc, int enemyIdx, int calculatedDamage) {
    assert(calculatedDamage >= 0);

    // assume intangible is already handled

    int damage = calculatedDamage;
    const auto savedBlock = block;
    block = std::max(0, block-damage);
    damage -= savedBlock;

    // buffer triggers before tungsten rod in the game's implementation
    // cases where tungsten rod would prevent damage // todo check if this is true
    if (damage > 0 && hasStatus<PS::BUFFER>()) {
        decrementStatus<PS::BUFFER>();
        damage = 0;
    }

    if (hasStatus<PS::THORNS>()) {
        Actions::DamageEnemy(bc, enemyIdx, getStatus<PS::THORNS>());
    }

    if (hasStatus<PS::FLAME_BARRIER>()) {
        Actions::DamageEnemy(bc, enemyIdx, getStatus<PS::FLAME_BARRIER>());
    }

    if (damage > 0 && damage <=5 && hasRelic<RelicId::TORII>()) {
        damage = 1;
    }

    if (damage > 0 && hasRelic<RelicId::TUNGSTEN_ROD>()) {
        damage -= 1;
    }

    if (damage > 0) {
        lastAttackUnblockedDamage = damage;

        if (hasStatus<PS::PLATED_ARMOR>()) {
            decrementStatus<PS::PLATED_ARMOR>();
        }

        if (bc.monsters.arr[enemyIdx].hasStatus<MS::PAINFUL_STABS>()) {
            Actions::MakeTempCardInDiscard(bc, {CardId::WOUND});
        }

        hpWasLost(bc, damage, false);

        if (hasStatus<PS::STATIC_DISCHARGE>()) {
            for (int i = 0; i < getStatus<PS::STATIC_DISCHARGE>(); ++i) {
                Actions::ChannelOrb(bc, Orb::LIGHTNING);
            }
        }

    } else {
        lastAttackUnblockedDamage = 0;
    }
}

void Player::loseHp(BattleContext &bc, int amount, bool selfDamage) {
    if (hasStatus<PS::INTANGIBLE>()) {
        amount = 1;
    }

    if (amount > 0 && hasRelic<RelicId::TUNGSTEN_ROD>()) {
        amount -= 1;
        if (amount == 0) {
            return;
        }
    }

    hpWasLost(bc, amount, selfDamage);
}

void Player::hpWasLost(BattleContext &bc, int amount, bool selfDamage) {
    assert(amount > 0);

    bool wasBloodied = curHp <= maxHp/2;

    curHp = std::max(0, curHp-amount);

    if (selfDamage && hasStatus<PS::RUPTURE>()) {
        buff<PS::STRENGTH>(getStatus<PS::RUPTURE>());
    }

    // todo - does order acquired matter with centennial/runic?
    // relics wasHpLost
    // -centennial-puzzle
    // -emotion chip
    // -runic cube
    // -self forming clay

    if (hasRelic<RelicId::CENTENNIAL_PUZZLE>()) {
        setHasRelic<RelicId::CENTENNIAL_PUZZLE>(false);
        Actions::DrawCards(bc, 3);
    }

    if (hasRelic<RelicId::EMOTION_CHIP>()) {
        // todo
    }

    if (hasRelic<RelicId::SELF_FORMING_CLAY>()) {
        buff<PS::NEXT_TURN_BLOCK>(3);
    }

    if (hasRelic<RelicId::RUNIC_CUBE>()) {
        Actions::DrawCards(bc, 1);
    }

    if (hasRelic<RelicId::RED_SKULL>() && !wasBloodied && curHp <= maxHp/2) {
        buff<PS::STRENGTH>(3);
    }

    bc.cards.onTookDamage();
    ++timesDamagedThisCombat;

    if (curHp <= 0) {
        wouldDie(bc);
    }
}

void Player::wouldDie(BattleContext &bc) {
    // assume fairy and lizard tail heal for greater than zero - max hp is not less than ~8
    curHp = 0;
    if  (!hasRelic<RelicId::MARK_OF_THE_BLOOM>()) {
        for (int i = 0; i < bc.potionCapacity; i++) {
            if (bc.potions[i] == Potion::FAIRY_POTION) {
                bc.discardPotion(i);
                const int healAmount = std::max(
                        1,
                        static_cast<int>(static_cast<float>(maxHp) * (hasRelic<R::SACRED_BARK>() ? 0.6f : 0.3f))
                );
                heal(healAmount);
                return;
            }
        }

        if (hasRelic<RelicId::LIZARD_TAIL>()) {
            setHasRelic<RelicId::LIZARD_TAIL>(false);
            heal(maxHp/2);
            return;
        }
    }

    bc.outcome = Outcome::PLAYER_LOSS;
}

void Player::applyEndOfTurnPowers(BattleContext &bc) {
    if (bomb1) {
        Actions::DamageAllEnemy(bc, bomb1);
    }
    bomb1 = bomb2;
    bomb2 = bomb3;
    bomb3 = 0;

    for (int i = 0; i < studyInsights; ++i) {
        Actions::MakeTempCardInDrawPile(bc, CardInstance(CardId::INSIGHT), 1, true);
    }

    for (auto pair : statusMap) {
        if (!hasStatusRuntime(pair.first)) {
            continue;
        }

        switch (pair.first) {
            case PS::BURST:
                Actions::RemoveStatus<PS::BURST>(bc);
                break;

            case PS::COMBUST:
                if (!bc.monsters.areMonstersBasicallyDead()) {
                    Actions::PlayerLoseHp(bc, combustHpLoss, true); // todo combust doesnt stack hp loss correctly
                    Actions::DamageAllEnemy(bc, pair.second);
                }
                break;

            case PS::CONSTRICTED:
                Actions::DamagePlayer(bc, pair.second);
                break;

            case PS::DOUBLE_TAP:
                Actions::RemoveStatus<PS::DOUBLE_TAP>(bc);
                break;

            case PS::ENTANGLED:
                Actions::RemoveStatus<PS::ENTANGLED>(bc);
                break;

            case PS::EQUILIBRIUM:
                // todo if card is ethereal set to retain
                break;

            case PS::LOOP:
                for (int i = 0; i < pair.second; ++i) {
                    triggerOrbEndOfTurn(bc, 0);
                }
                break;

            case PS::ESTABLISHMENT:
                break;

            case PS::LOSE_DEXTERITY:
                Actions::DebuffPlayer<PS::DEXTERITY>(bc, -pair.second);
                Actions::RemoveStatus<PS::LOSE_DEXTERITY>(bc);
                break;

            case PS::LOSE_STRENGTH:
                Actions::DebuffPlayer<PS::STRENGTH>(bc, -pair.second);
                Actions::RemoveStatus<PS::LOSE_STRENGTH>(bc);
                break;

            case PS::NO_DRAW:
                Actions::RemoveStatus<PS::NO_DRAW>(bc);
                break;

            case PS::OMEGA:
                Actions::DamageAllEnemy(bc, pair.second);
                break;

            case PS::RAGE:
                removeStatus<PS::RAGE>();
                break;

            case PS::REBOUND:
                Actions::RemoveStatus<PS::REBOUND>(bc);
                break;

            case PS::REGEN:
                Actions::HealPlayer(bc, pair.second);
                Actions::DecrementStatus<PS::REGEN>(bc);
                break;


            case PS::RITUAL:
                Actions::BuffPlayer<PS::STRENGTH>(bc, pair.second);
                break;
                // case TheBomb

            case PS::WRAITH_FORM: // todo does this debuff or just decrement?
                Actions::DecrementStatus<PS::DEXTERITY>(bc, pair.second);
                break;

            default:
                break;
        }
    }
}

void Player::applyAtEndOfRoundPowers() {
    if (hasStatus<PS::FRAIL>()) {
        decrementIfNotJustApplied<PS::FRAIL>();
    }

    if (hasStatus<PS::VULNERABLE>()) {
        decrementIfNotJustApplied<PS::VULNERABLE>();
    }

    if (hasStatus<PS::WEAK>()) {
        decrementIfNotJustApplied<PS::WEAK>();
    }

    // handling this later so it is not removed before block check
//    if (hasStatusInternal<PS::BLUR>()) {
//        decrementStatus<PS::BLUR>();
//    }

    if (hasStatus<PS::DOUBLE_DAMAGE>()) {
        decrementIfNotJustApplied<PS::DOUBLE_DAMAGE>();
    }

    // handle elsewhere
//    if (hasStatusInternal<PS::DRAW_REDUCTION>()) {
//        decrementStatus<PS::DRAW_REDUCTION>();
//    }

    if (hasStatus<PS::DUPLICATION>()) {
        decrementStatus<PS::DUPLICATION>();
    }

    if (hasStatus<PS::EQUILIBRIUM>()) {
        decrementStatus<PS::EQUILIBRIUM>();
    }

    if (hasStatus<PS::INTANGIBLE>()) {
        decrementStatus<PS::INTANGIBLE>();
    }

    if (hasStatus<PS::NO_BLOCK>()) {
        decrementStatus<PS::NO_BLOCK>();
    }

    if (hasStatus<PS::WAVE_OF_THE_HAND>()) {
        decrementStatus<PS::WAVE_OF_THE_HAND>();
    }
}

void Player::applyStartOfTurnRelics(BattleContext &bc) {
    //****** Player relics atTurnStart ******
    if (hasRelic<R::ART_OF_WAR>()) {
        if (attacksPlayedThisTurn == 0) {
            Actions::GainEnergy(bc, 1);
        }
    }

    if (hasRelic<R::BRIMSTONE>()) {
        buff<PS::STRENGTH>(2);
        for (int i = 0; i < bc.monsters.monsterCount; i++) {
            if (bc.monsters.arr[i].isTargetable()) {
                bc.monsters.arr[i].buff<MS::STRENGTH>(1);
            }
        }
    }

    if (hasRelic<R::CAPTAINS_WHEEL>()) {
        if (bc.turn == 2) {
            Actions::GainBlock(bc, 18);
        }
    }

    if (hasRelic<R::DAMARU>()) {
        Actions::BuffPlayer<PS::MANTRA>(bc, 1);
        // todo handle mantra change stance
    }

    if (hasRelic<R::EMOTION_CHIP>()) {
    }

    if (hasRelic<R::HAPPY_FLOWER>()) {
        if (++happyFlowerCounter == 3) {
            happyFlowerCounter = 0;
            Actions::GainEnergy(bc, 1);
        }
    }

    if (hasRelic<R::HORN_CLEAT>()) {
        if (bc.turn == 1) {
            Actions::GainBlock(bc, 14);
        }
    }

    if (hasRelic<R::INCENSE_BURNER>()) {
        if (++incenseBurnerCounter == 6) {
            incenseBurnerCounter = 0;
            Actions::BuffPlayer<PS::INTANGIBLE>(bc, 1);
        }
    }

    if (hasRelic<R::INSERTER>()) {
        if (++inserterCounter == 2) {
            inserterCounter = 0; // todo
            increaseOrbSlots(1);
        }
    }

    if (hasRelic<R::MERCURY_HOURGLASS>()) {
        Actions::DamageAllEnemy(bc, 3);
    }

    if (hasRelic<R::NECRONOMICON>()) {
        haveUsedNecronomiconThisTurn = false;
    }

    if (hasRelic<R::ORANGE_PELLETS>()) {
        orangePelletsCardTypesPlayed.reset();
    }

}

void Player::applyStartOfTurnPowers(BattleContext &bc) {
    // ****** Player powers atStartOfTurn ******
    for (int i = 0; i < bc.nightmareCards.size(); ++i) {
        for (int copy = 0; copy < bc.nightmareCopyCounts[i]; ++copy) {
            bc.moveToHandHelper(bc.nightmareCards[i]);
        }
    }
    bc.nightmareCards.clear();
    bc.nightmareCopyCounts.clear();

    for (auto pair : statusMap) {
        if (!hasStatusRuntime(pair.first)) {
            continue;
        }

        switch (pair.first) {
            case PS::BATTLE_HYMN:
                Actions::MakeTempCardInHand(bc, CardId::SMITE, hasStatus<PS::MASTER_REALITY>(), pair.second);
                break;

            case PS::BIAS:
                Actions::DecrementStatus<PS::FOCUS>(bc, pair.second);
                break;

            case PS::CREATIVE_AI:
//                Actions::SetState(bc, InputState::CREATE_RANDOM_CARD_IN_HAND_POWER, pair.second); // todo
                break;

            case PS::ECHO_FORM:
                echoFormCardsDoubled = 0;
                break;

            case PS::BLASPHEMER:
                removeStatus<PS::BLASPHEMER>();
                Actions::PlayerLoseHp(bc, 9999);
                break;

            case PS::FASTING:
                //  add energy dont need? just change energy per turn instead
                break;

            case PS::FORESIGHT:
                if (bc.cards.drawPile.empty() && !bc.cards.discardPile.empty()) {
                    bc.onShuffle();
                    Actions::EmptyDeckShuffle(bc);
                }
                Actions::ScryAction(bc, pair.second);
                break;

            case PS::FLAME_BARRIER:
                removeStatus<PS::FLAME_BARRIER>();
                break;

            case PS::HELLO_WORLD: // todo
                removeStatus<PS::HELLO_WORLD>();
                break;

            case PS::INFINITE_BLADES:
                Actions::MakeTempCardInHand(bc, CardId::SHIV, hasStatus<PS::MASTER_REALITY>(), pair.second);
                break;

            case PS::LOOP:
                for (int i = 0; i < pair.second; ++i) {
                    triggerOrbStartOfTurn(bc, 0);
                }
                break;

            case PS::MAGNETISM:
//                Actions::SetState(bc, InputState::CREATE_RANDOM_CARD_IN_HAND_COLORLESS, pair.second);
                break;

            case PS::MAYHEM:
                for (int i = 0; i < pair.second; i++) {
                    Actions::PlayTopCard(bc, bc.monsters.getRandomMonsterIdx(bc.cardRandomRng), false); // todo fix target
                }
                break;

            case PS::NEXT_TURN_BLOCK:
                Actions::GainBlock(bc, pair.second);
                removeStatus<PS::NEXT_TURN_BLOCK>();
                break;


            case PS::PANACHE:
                panacheCounter = 5;
                break;

            case PS::PHANTASMAL:
                decrementStatus<PS::PHANTASMAL>();
                Actions::BuffPlayer<PS::DOUBLE_DAMAGE>(bc);
                break;

                // time maze not used in standard modes

            case PS::WRATH_NEXT_TURN:
                removeStatus<PS::WRATH_NEXT_TURN>();
                Actions::ChangeStance(bc, Stance::WRATH);
                break;

            default:
                break;
        }

    }

    if (creativeAi > 0) {
        Actions::MakeRandomCardInHand(bc, CardType::POWER, creativeAi, true);
    }
    if (helloWorld > 0) {
        Actions::MakeRandomCommonCardInHand(bc, helloWorld);
    }
}

void Player::applyStartOfTurnPostDrawRelics(BattleContext &bc) {
    // ****** Player Relics AtTurnStartPostDraw ******
    if (hasRelic<R::POCKETWATCH>()) {
        if (cardsPlayedThisTurn <= 3) {
            Actions::DrawCards(bc, 3);
        }
    }

    if (hasRelic<R::WARPED_TONGS>()) {
        Actions::UpgradeRandomCardAction(bc);
    }
}

void Player::applyStartOfTurnPostDrawPowers(BattleContext &bc) {
    // ****** Player Powers AtStartOfTurnPostDraw ******
    for (auto pair :statusMap) {
        if (!hasStatusRuntime(pair.first)) {
            continue;
        }

        switch (pair.first) {
            case PS::BRUTALITY:
                Actions::PlayerLoseHp(bc, pair.second);
                Actions::DrawCards(bc, pair.second);
                break;

            case PS::DEMON_FORM:
                Actions::BuffPlayer<PS::STRENGTH>(bc, pair.second);
                break;

            case PS::DEVOTION: // the implementation of this is really weird in the game code
                Actions::BuffPlayer<PS::MANTRA>(bc, pair.second); // todo make buffing mantra switch stance
                break;

            case PS::DRAW_CARD_NEXT_TURN:
                Actions::DrawCards(bc, pair.second);
                removeStatus<PS::DRAW_CARD_NEXT_TURN>();
                break;

            case PS::NOXIOUS_FUMES:
                Actions::DebuffAllEnemy<MS::POISON>(bc, pair.second);
                break;

            case PS::TOOLS_OF_THE_TRADE:
                Actions::DrawCards(bc, pair.second);
                break;

            default:
                break;
        }
    }

    if (machineLearning > 0) {
        Actions::DrawCards(bc, machineLearning);
    }
}

void Player::rechargeEnergy(BattleContext &bc) {
    if (hasRelic<R::ICE_CREAM>()) {
        gainEnergy(energyPerTurn);
    } else {
        energy = energyPerTurn;
    }

    // ****** Player powers onEnergyRecharge ******
    if (hasStatus<PS::COLLECT>()) {
        decrementStatus<PS::COLLECT>();
        Actions::MakeTempCardInHand(bc, CardId::MIRACLE);
    }

    if (hasStatus<PS::DEVA>()) {
        gainEnergy(devaFormEnergyPerTurn);
        devaFormEnergyPerTurn += getStatus<PS::DEVA>();
    }

    if (hasStatus<PS::ENERGIZED>()) {
        gainEnergy(getStatus<PS::ENERGIZED>());
        removeStatus<PS::ENERGIZED>();
    }
}

std::ostream& printIfHaveStatus(const Player &p, std::ostream &os, PlayerStatus s) {
    if (!p.hasStatusRuntime(s)){
        return os;
    }

    std::string desc = playerStatusStrings[(int)s];
    return os << "(" << desc << "," << p.getStatusRuntime(s) << ")" << ", ";
}



//    void printRelicCounters(std::ostream &os, const Player &p) {
//        const std::string s = " ";
//
//        if (p.hasRelic<R::INK_BOTTLE>()) {
//            os << s << "inkBottleCounter: " << p.inkBottleCounter;
//        }
//
//        if (p.hasRelic<R::INSERTER>()) {
//            os << "inserterCounter: " << p.inserterCounter << " ";
//        }
//
//        if (p.hasRelic<R::NUNCHAKU>()) {
//            os << "nunchakuCounter: " << p.nunchakuCounter << " ";
//        }
//
//        if (p.hasRelic<R::LETTER_OPENER>()) {
//            os << "skillsPlayedThisTurn: " << p.skillsPlayedThisTurn << " ";
//        }
//
//    }
//
//    void printStatusCounters(std::ostream &os, const Player &p) {
//
//        if (p.hasStatusInternal<PS::COMBUST>()) {
//            os << "combustHpLoss: " << p.combustHpLoss << " ";
//        }
//
//        if (p.hasStatusInternal<PS::DEVA>()) {
//            os << "devaFormEnergyPerTurn: " << p.devaFormEnergyPerTurn << " ";
//        }
//
//        if (p.hasStatusInternal<PS::ECHO_FORM>()) {
//            os << "panacheCounter: " << p.panacheCounter << " ";
//        }
//
//
//        if (p.hasStatusInternal<PS::PANACHE>()) {
//            os << "panacheCounter: " << p.panacheCounter << " ";
//        }
//    }

    void printRelics(std::ostream &os, const Player &p) {
        os << "\t" << "Relics: ";

        // this is slow and stupid
        int i = static_cast<int>(R::AKABEKO);
        while (i <= static_cast<int>(R::WRIST_BLADE)) {
            auto r = static_cast<RelicId>(i);

            if (p.hasRelicRuntime(r)) {
                os << getRelicName(r) << ", ";
            }

            ++i;
        }
        os << "\n";
    }

    void printAllInfos(std::ostream &os, const Player &p) {
        const std::string s = ", ";

        os << "\t" << "Relic Counters: "
            << "" << "happyFlowerCounter: " << static_cast<int>(p.happyFlowerCounter)
            << s << "incenseBurnerCounter: " << static_cast<int>(p.incenseBurnerCounter)
            << s << "inkBottleCounter: " << static_cast<int>(p.inkBottleCounter)
            << s << "inserterCounter: " << static_cast<int>(p.inserterCounter)
            << s << "nunchakuCounter: " << static_cast<int>(p.nunchakuCounter)
            << s << "penNibCounter: " << static_cast<int>(p.penNibCounter)
            << s << "sundialCounter: " << static_cast<int>(p.sundialCounter)

            << "\n\t" << "Status Counters: "
            << "" << "combustHpLoss: " << static_cast<int>(p.combustHpLoss)
            << s << "devaFormEnergyPerTurn: " << static_cast<int>(p.devaFormEnergyPerTurn)
            << s << "echoFormCardsDoubled: " << static_cast<int>(p.echoFormCardsDoubled)
            << s << "panacheCounter: " << static_cast<int>(p.panacheCounter)
            << s << "TheBomb: { 1=" << static_cast<int>(p.bomb1)
                << ", 2=" << static_cast<int>(p.bomb2)
                << ", 3=" << static_cast<int>(p.bomb3)
                << " }"

            << "\n\t" << "Misc: "
            << "" << "cardsPlayedThisTurn: " << static_cast<int>(p.cardsPlayedThisTurn)
            << s << "attacksPlayedThisTurn: " << static_cast<int>(p.attacksPlayedThisTurn)
            << s << "skillsPlayedThisTurn: " << static_cast<int>(p.skillsPlayedThisTurn);


            os << s << "orangePelletsCardTypesPlayed: {"
                << (p.orangePelletsCardTypesPlayed.test(0) ? "Attack," : "")
                << (p.orangePelletsCardTypesPlayed.test(1) ? "Skill," : "")
                << (p.orangePelletsCardTypesPlayed.test(2) ? "Power," : "")
                << "}";

            os << s << "cardsDiscardedThisTurn: " << static_cast<int>(p.cardsDiscardedThisTurn);
            os << s << "gold: " << static_cast<int>(p.gold) << '\n';
    }


    void printStatusEffects(std::ostream &os, const Player &p) {
        os << "\tStatusEffects: {";
        if (p.hasStatus<PS::CONFUSED>()) {
            os << "(Confused), ";
        }
        if (p.hasStatus<PS::CORRUPTION>()) {
            os << "(Corruption), ";
        }
        if (p.hasStatus<PS::BARRICADE>()) {
            os << "(Barricade), ";
        }

        printIfHaveStatus(p, os, PS::ARTIFACT);
        printIfHaveStatus(p, os, PS::DEXTERITY);
        printIfHaveStatus(p, os, PS::FOCUS);
        printIfHaveStatus(p, os, PS::STRENGTH);
        for (auto pair : p.statusMap) {
            printIfHaveStatus(p, os, pair.first);
        }
        os << "}\n";
    }

    std::ostream& operator<<(std::ostream &os, const Player &p) {
        os << "Player: {\n";

        os << "\t" << "hp:(" << p.curHp << "/" << p.maxHp << ")"
            << " energy:(" << p.energy << "/" << static_cast<int>(p.energyPerTurn)
        << ") block:(" << p.block << ")\n";

        printStatusEffects(os, p);
        printRelics(os, p);
        printAllInfos(os, p);

        os << "}\n";
        return os;
    }



}












