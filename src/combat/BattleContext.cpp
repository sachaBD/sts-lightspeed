//
// Created by gamerpuppy on 7/4/2021.
//

#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "game/Game.h"
#include <stdexcept>

namespace sts {
    thread_local BattleContext *g_debug_bc;


// assume all bc fields have just been initialized by in class member initializers
void BattleContext::init(const GameContext &gc) {
    init(gc, gc.info.encounter);
}

void BattleContext::init(const GameContext &gc, MonsterEncounter encounterToInit) {
    undefinedBehaviorEvoked = false;
    haveUsedDiscoveryAction = false;
    seed = gc.seed;
    floorNum = gc.floorNum;
    encounter = encounterToInit;

    auto startRandom = Random(gc.seed+gc.floorNum);
    aiRng = startRandom;
    monsterHpRng = startRandom;
    shuffleRng = startRandom;
    cardRandomRng = startRandom;
    miscRng = gc.miscRng;
    potionRng = gc.potionRng;

    ascension = gc.ascension;
    outcome = Outcome::UNDECIDED;
    inputState = InputState::EXECUTING_ACTIONS;
    miscBits.reset();

    monsterTurnIdx = 6;
    skipMonsterTurn = false;
    endTurnAfterCurrentCard = false;
    turnHasEnded = false;
    isBattleOver = false;

    resolution.clear();
    endTurnRequested = false;

    potionCount = gc.potionCount;
    potionCapacity = gc.potionCapacity;
    potions = gc.potions;

    player.curHp = gc.curHp;
    player.maxHp = gc.maxHp;
    player.gold = gc.gold;
    player.cc = gc.cc;
    player.orbSlots = gc.cc == CharacterClass::DEFECT ? 3 : 0;
    player.orbs.fill(Orb::EMPTY);
    player.orbData.fill(0);

    monsters.init(*this, encounterToInit);
    if (gc.map->burningEliteX == gc.curMapNodeX && gc.map->burningEliteY == gc.curMapNodeY) {
        monsters.applyEmeraldEliteBuff(*this, gc.map->burningEliteBuff, gc.act);
    }

    player.cardDrawPerTurn = 5;
    if (gc.hasRelic(R::SNECKO_EYE)) {
        player.cardDrawPerTurn += 2;
    }
    if (gc.relics.has(R::RING_OF_THE_SERPENT)) {
        player.cardDrawPerTurn += 1;
    }
    //Actions::DrawCards(*this, player.cardDrawPerTurn);

    cards.init(gc, *this);

    initRelics(gc);
    player.energy += player.energyPerTurn;

    resolution.push_back({ResolutionPhase::InitialToolbox});
    executeActions();
}

// this doesnt apply powers in order, so if that matters in the future all relics will have to be sorted
void BattleContext::initRelics(const GameContext &gc) {
    player.relicBits0 = gc.relics.relicBits0;
    player.relicBits1 = gc.relics.relicBits1;

    fixed_list<RelicId, 4> atBattleStartPreDraw;
    fixed_list<RelicId, 8> atBattleStart;
    fixed_list<RelicId, 2> atTurnStartPostDraw;

    auto room = gc.curRoom;

    auto &p = player;

    for (const auto &r : gc.relics.relics) {
        switch (r.id) {

            case R::HOLY_WATER:
            case R::NINJA_SCROLL:
            case R::PURE_WATER:
            case R::TOOLBOX:
                atBattleStartPreDraw.push_back(r.id);
                break;

            case R::BAG_OF_MARBLES:
            case R::BAG_OF_PREPARATION:
            case R::CLOCKWORK_SOUVENIR:
            case R::GREMLIN_VISAGE:
            case R::RED_MASK:
            case R::RING_OF_THE_SNAKE:
            case R::TWISTED_FUNNEL:
                atBattleStart.push_back(r.id);
                break;

            case R::MARK_OF_PAIN:
                ++p.energyPerTurn;
                atBattleStart.push_back(r.id);
                break;

            case R::GAMBLING_CHIP:
            case R::WARPED_TONGS:
                atTurnStartPostDraw.push_back(r.id);
                break;

            case R::AKABEKO:
                p.buff<PS::VIGOR>(8);
                break;

            case R::BRIMSTONE:
                p.buff<PS::STRENGTH>(2);
                for (int i = 0; i < monsters.monsterCount; ++i) {
                    Monster &m = monsters.arr[i];
                    if (m.isTargetable()) {
                        m.buff<MS::STRENGTH>(1);
                    }
                }
                break;

            case R::ECTOPLASM:
                p.energyPerTurn++;
                break;

            case R::ENCHIRIDION: {
                const auto cardId = getTrulyRandomCardInCombat(cardRandomRng, p.cc, CardType::POWER);
                CardInstance c(cardId);
                c.setCostForTurn(0);
                Actions::MakeTempCardInHand(*this, c);
                break;
            }

            case R::HAPPY_FLOWER:
                player.happyFlowerCounter = r.data + 1;
                if (player.happyFlowerCounter == 3) {
                    ++player.energy;
                    player.happyFlowerCounter = 0;
                }
                break;

            case R::INCENSE_BURNER:
                p.incenseBurnerCounter = r.data;
                if (++p.incenseBurnerCounter == 6) {
                    p.incenseBurnerCounter = 0;
                    p.buff<PS::INTANGIBLE>(1);
                }
                break;

            case R::INK_BOTTLE:
                p.inkBottleCounter = r.data;
                break;

            case R::INSERTER:
                if (r.data) {
                    p.inserterCounter = 0;
                    p.increaseOrbSlots(1);
                } else {
                    p.inserterCounter = 1;
                }
                break;

            case R::LIZARD_TAIL:
                p.setHasRelic<R::LIZARD_TAIL>(r.data);
                break;

            case R::NUNCHAKU:
                p.nunchakuCounter = r.data;
                break;

            case R::OMAMORI:
                p.setHasRelic<R::OMAMORI>(r.data);
                break;

            case R::PEN_NIB:
                if (r.data == 9) {
                    p.buff<PS::PEN_NIB>(1);
                    p.penNibCounter = -1;
                } else {
                    p.penNibCounter = r.data;
                }
                break;

            case R::PHILOSOPHERS_STONE:
                for (int i = 0; i < monsters.monsterCount; ++i) {
                    auto &m = monsters.arr[i];
                    m.buff<MS::STRENGTH>(1);
                }
                p.energyPerTurn++;
                break;

            case R::RUNIC_DOME:
                p.energyPerTurn++;
                break;

            case R::SNECKO_EYE:
                p.debuff<PS::CONFUSED>(1);
                break;

            case R::SOZU:
                p.energyPerTurn++;
                break;

            case R::SUNDIAL:
                p.sundialCounter = r.data;
                break;

            case R::VELVET_CHOKER:
                p.energyPerTurn++;
                break;

            case R::ANCHOR:
                p.block += 10;
                break;

            case R::ANCIENT_TEA_SET:
                if (gc.lastRoom == Room::REST) {
                    p.gainEnergy(2);
                }
                break;

            case R::BLOOD_VIAL:
                p.heal(2); // todo not correct
                break;

            case R::BRONZE_SCALES:
                p.buff<PS::THORNS>(3);
                break;

            case R::BUSTED_CROWN:
                p.energyPerTurn++;
                break;

            case R::COFFEE_DRIPPER:
                p.energyPerTurn++;
                break;

            case R::CRACKED_CORE:
                p.channelOrb(*this, Orb::LIGHTNING);
                break;

            case R::CURSED_KEY:
                p.energyPerTurn++;
                break;

            case R::DAMARU:
                p.buff<PS::MANTRA>(1);
                break;

            case R::DATA_DISK:
                p.buff<PS::FOCUS>(1);
                break;

            case R::DU_VU_DOLL:
                p.buff<PS::STRENGTH>(r.data);
                break;

            case R::FOSSILIZED_HELIX:
                p.buff<PS::BUFFER>(1);
                break;

            case R::FUSION_HAMMER:
                p.energyPerTurn++;
                break;

            case R::GIRYA:
                p.buff<PS::STRENGTH>(r.data);
                break;

            case R::LANTERN:
                p.gainEnergy(1);
                break;

            case R::MUTAGENIC_STRENGTH: // this appears to be applied before clockwork if it was acquired first
                p.buff<PS::STRENGTH>(3);
                p.debuff<PS::LOSE_STRENGTH>(3);
                break;

            case R::NEOWS_LAMENT: // remember to decrement somewhere else
                if (r.data > 0) {
                    for (int i = 0; i < monsters.monsterCount; ++i) {
                        Monster &m = monsters.arr[i];
                        m.curHp = 1;
                    }
                }
                break;

            case R::NUCLEAR_BATTERY:
                p.channelOrb(*this, Orb::FUSION);
                break;

            case R::ODDLY_SMOOTH_STONE:
                p.buff<PS::DEXTERITY>(1);
                break;

            case R::PANTOGRAPH:
                if (room == Room::BOSS) {
                    p.heal(25);
                }
                break;

            case R::PRESERVED_INSECT:
                if (room == Room::ELITE) {
                    for (int i = 0; i < monsters.monsterCount; ++i) {
                        Monster &m = monsters.arr[i];
                        m.curHp = static_cast<int>(m.maxHp * .75);
                    }
                }
                break;

            case R::RING_OF_THE_SERPENT:
                // now handled in battlecontext init
//                p.cardDrawPerTurn++;
                break;

            case R::RUNIC_CAPACITOR:
                p.increaseOrbSlots(3);
                break;

            case R::SLAVERS_COLLAR:
                if (room == Room::ELITE || room == Room::BOSS) { // todo this needs to be set by eliteTrigger maybe?
                    p.energyPerTurn++;
                }
                break;

            case R::SLING_OF_COURAGE:
                if (room == Room::ELITE) {
                    p.buff<PS::STRENGTH>(2);
                }
                break;

            case R::SYMBIOTIC_VIRUS:
                p.channelOrb(*this, Orb::DARK);
                break;

            case R::TEARDROP_LOCKET:
                p.changeStance<Stance::CALM>();
                break;

            case R::THREAD_AND_NEEDLE:
                p.buff<PS::PLATED_ARMOR>(4);
                break;

            case R::VAJRA:
                p.buff<PS::STRENGTH>(1);
                break;

            default:
                break;
        }
    }

    // todo maybe move this to proper place -nvm drawCards is added below this
    for (auto r : atBattleStartPreDraw) {
        switch (r) {
            case R::HOLY_WATER:
                Actions::MakeTempCardInHand(*this, CardId::MIRACLE, false, 3);
                break;

            case R::NINJA_SCROLL:
                Actions::MakeTempCardInHand(*this, CardId::SHIV, false, 3);
                break;

            case R::PURE_WATER:
                Actions::MakeTempCardInHand(*this, CardId::MIRACLE, false, 1);
                break;

            case R::TOOLBOX:
                break;

            default:
                break;
        }
    }

    Actions::DrawCards(*this, p.cardDrawPerTurn);

    for (auto r : atBattleStart) {
        switch (r) {
            case R::BAG_OF_MARBLES:
                Actions::DebuffAllEnemy<MS::VULNERABLE>(*this, 1, false);
                break;

            case R::BAG_OF_PREPARATION:
                Actions::DrawCards(*this, 2);
                break;

            case R::CLOCKWORK_SOUVENIR:
                Actions::BuffPlayer<PS::ARTIFACT>(*this, 1);
                break;

            case R::GREMLIN_VISAGE:
                p.debuff<PS::WEAK>(1);
                break;

            case R::MARK_OF_PAIN:
                Actions::MakeTempCardInDrawPile(*this,  {CardId::WOUND}, 2, true);
                break;

            case R::RED_MASK:
                Actions::DebuffAllEnemy<MS::WEAK>(*this, 1);
                break;

            case R::RING_OF_THE_SNAKE:
                Actions::DrawCards(*this, 2);
                break;

            case R::TWISTED_FUNNEL:
                Actions::DebuffAllEnemy<MS::POISON>(*this, 4);
                break;

            default:
                break;
        }
    }

    if (gc.hasRelic(R::MERCURY_HOURGLASS)) {
        Actions::DamageAllEnemy(*this, 3);
    }

    if (gc.hasRelic(R::RED_SKULL) && gc.curHp <= gc.maxHp / 2) {
        p.buff<PS::STRENGTH>(3);
    }

    for (auto r : atTurnStartPostDraw) {
        switch (r) {
            case R::GAMBLING_CHIP:
                break;

            case R::WARPED_TONGS:
                Actions::UpgradeRandomCardAction(*this);
                break;

            default:
                break;
        }
    }

    // ** OnStartOfTurn ** ORBS todo
    //RelicCables -> OnStartOfTurn again for orb 0
}

void BattleContext::exitBattle(GameContext &g) const {
    // do this first so that darkstone periapt is overridden by curHp and maxHp are set afterwards
    const auto &m = monsters.arr[0];
    if (m.id == MonsterId::WRITHING_MASS && m.miscInfo) {
        if (player.hasRelic<R::OMAMORI>()) {
            --g.relics.getRelicValueRef(RelicId::OMAMORI);
        } else {
            g.deck.obtain(g, CardId::PARASITE);

        }
    }

    g.potionCount = potionCount;
    g.potions = potions;

    // not sure its really necessary to sync these every time, (i believe colosseum is the only time two battles occur on the same floor)
    g.aiRng = aiRng;
    g.cardRandomRng = cardRandomRng;
    g.miscRng = miscRng;
    g.monsterHpRng = monsterHpRng;
    g.potionRng = potionRng;
    g.shuffleRng = shuffleRng;

    g.curHp = player.curHp;
    g.maxHp = player.maxHp;
    g.gold = player.gold;

    if (escapedCombat) {
        // Smoke Bomb returns to the map without combat rewards, victory-only
        // healing, or post-combat relic effects.
        g.screenState = ScreenState::MAP_SCREEN;
        g.regainControlAction = nullptr;
        return;
    }

    if (outcome == Outcome::PLAYER_VICTORY && player.selfRepairHeal > 0) {
        g.playerHeal(player.selfRepairHeal);
    }


    // todo lesson learned bitset

    // relic counters
    updateRelicsOnExit(g);

    // cards
    updateCardsOnExit(g.deck);

    g.info.stolenGold = 0;
    if (requiresStolenGoldCheck()) {
        for (int i = 0; i < monsters.monsterCount; ++i) {
            const auto &m = monsters.arr[i];

            const bool canHaveStolenGold = m.id == MonsterId::LOOTER || m.id == MonsterId::MUGGER;
            const bool escaped = m.curHp > 0 && (m.moveHistory[0] == MMID::LOOTER_ESCAPE ||
                                                 m.moveHistory[0] == MMID::MUGGER_ESCAPE);

            if (canHaveStolenGold && !escaped) {
                g.info.stolenGold += m.miscInfo;
            }
        }
    }

    if (outcome == Outcome::PLAYER_LOSS) {
        g.outcome = GameOutcome::PLAYER_LOSS;
    } else {
        // player victory
        g.regainControl();
    }

    BattleContext::sum += g.curHp + g.maxHp + g.gold + g.act
            + g.ascension + g.floorNum + potionRng.counter + cardRandomRng.counter;
}

void BattleContext::updateRelicsOnExit(GameContext &g) const {
    for (auto &r : g.relics.relics) {
        switch (r.id) {
            case RelicId::HAPPY_FLOWER:
                r.data = player.happyFlowerCounter;
                break;

            case RelicId::INCENSE_BURNER:
                r.data = player.incenseBurnerCounter;
                break;

            case RelicId::INK_BOTTLE:
                r.data = player.inkBottleCounter;
                break;

            case RelicId::INSERTER:
                r.data = player.inserterCounter;
                break;

            case RelicId::NEOWS_LAMENT:
                if (r.data > 0) {
                    --r.data;
                }
                break;

            case RelicId::NUNCHAKU:
                r.data = player.nunchakuCounter;
                break;

            case RelicId::PEN_NIB:
                // possible bug
                if (player.penNibCounter == -1) {
                    r.data = 9;
                } else {
                    r.data = player.penNibCounter;
                }
                break;

            case RelicId::SUNDIAL:
                r.data = player.sundialCounter;
                break;

            case RelicId::LIZARD_TAIL:
                if (!player.hasRelic<R::LIZARD_TAIL>()) {
                    r.data = 0;
                }
                break;

            case RelicId::BURNING_BLOOD:
                if (outcome == Outcome::PLAYER_VICTORY) {
                    g.playerHeal(6);
                }
                break;

            case RelicId::BLACK_BLOOD:
                if (outcome == Outcome::PLAYER_VICTORY) {
                    g.playerHeal(12);
                }
                break;

            case RelicId::MEAT_ON_THE_BONE:
                if (outcome == Outcome::PLAYER_VICTORY && g.curHp <= g.maxHp / 2) {
                    g.playerHeal(12);
                }
                break;

            default:
                break;
        }

    }
}

void cardOnExit(const CardInstance &c, Deck &deck) {
    const auto isDeckCard = c.uniqueId < deck.size();
    if (!isDeckCard) {
        return;
    }

    const auto deckIdx = c.uniqueId;
    auto &deckCard = deck.cards[deckIdx];

    if (c.getId() == CardId::RITUAL_DAGGER) {
        deckCard.misc = c.specialData;
    }

    // todo
//    if (c.getId() == CardId::GENETIC_ALGORITHM) {
//        deckCard.misc = c.specialData;
//    }
}

void BattleContext::updateCardsOnExit(Deck &deck) const {
    for (const auto &c : cards.drawPile) {
        cardOnExit(c, deck);
    }

    for (const auto &c : cards.discardPile) {
        cardOnExit(c, deck);
    }

    for (const auto &c : cards.exhaustPile) {
        cardOnExit(c, deck);
    }

    for (int i = 0; i < cards.cardsInHand; ++i) {
        cardOnExit(cards.hand[i], deck);
    }

//    if (currentCard)

}


void BattleContext::setRequiresStolenGoldCheck(bool value) {
    miscBits.set(0, value);
}

bool BattleContext::requiresStolenGoldCheck() const {
    return miscBits.test(0);
}

int BattleContext::getMonsterTurnNumber() const {
    return turn+1; // todo;
}

void BattleContext::setState(InputState s) {
    inputState = s;
}

void BattleContext::beginCard(CardPlay item) {
    if (outcome == Outcome::UNDECIDED) resolution.push_back({ResolutionPhase::CardPrepare, item});
}

void BattleContext::checkCombat() {
    if (outcome != Outcome::PLAYER_VICTORY) return;
    for (auto& frame : resolution) {
        if (!frame.isCard()) frame.phase = ResolutionPhase::Done;
        else if (frame.phase == ResolutionPhase::CardPrepare) frame.phase = ResolutionPhase::CardDone;
        else if (frame.phase != ResolutionPhase::CardDone) frame.phase = ResolutionPhase::CardCleanup;
    }
}

bool BattleContext::isCardPlayAllowed() const {
    if (player.hasRelic<R::VELVET_CHOKER>() && player.cardsPlayedThisTurn >= 6) {
        return false;
    }

    if (cards.handNormalityCount && player.cardsPlayedThisTurn >= 3) {
        return false;
    }

    return true;
}

void BattleContext::executeActions() {
    ++sum;
    g_debug_bc = this;
    using P = ResolutionPhase;
    for (;;) {
        if (++loopCount > 1000000 || turn > 500) throw std::runtime_error{"combat resolution limit"};
        if (outcome == Outcome::PLAYER_LOSS) { resolution.clear(); return; }
        if (inputState != InputState::EXECUTING_ACTIONS && outcome == Outcome::UNDECIDED) return;
        if (resolution.empty()) {
            if (outcome != Outcome::UNDECIDED) return;
            if (endTurnRequested) {
                endTurnRequested = false;
                resolution.push_back({P::EndRelics});
                continue;
            }
            if (player.hasRelic<R::UNCEASING_TOP>() && cards.cardsInHand == 0) drawCards(1);
            if (outcome != Outcome::UNDECIDED) return;
            const bool noCards = cards.cardsInHand + cards.drawPile.size() + cards.discardPile.size() == 0;
            const bool otherDamage = player.hasStatus<PS::OMEGA>() || player.hasStatus<PS::THORNS>() ||
                player.hasStatus<PS::COMBUST>() || player.hasStatus<PS::JUGGERNAUT>() ||
                player.bomb1 || player.bomb2 || player.bomb3;
            if (noCards && !otherDamage && monsters.arr[0].id != MonsterId::TRANSIENT)
                outcome = Outcome::PLAYER_LOSS;
            inputState = InputState::PLAYER_NORMAL;
            return;
        }
        const auto index = resolution.size() - 1;
        const auto phase = resolution[index].phase;
        const bool card = resolution[index].isCard();
        if (card) currentCard = resolution[index].card;
        // Advance before calling effects: a nested card may push another frame or request a choice.
        switch (phase) {
            case P::InitialToolbox:
                resolution[index].phase = P::InitialGamble;
                if (player.hasRelic<R::TOOLBOX>()) Actions::ToolboxAction(*this);
                break;
            case P::InitialGamble:
                resolution[index].phase = P::Done;
                if (player.hasRelic<R::GAMBLING_CHIP>()) Actions::GambleAction(*this);
                break;
            case P::CardPrepare:
                resolution[index].phase = prepareCard() ? P::CardEffects : P::CardDone;
                break;
            case P::CardEffects:
                resolution[index].phase = P::CardAfterChoice;
                if (outcome != Outcome::UNDECIDED) break;
                switch (currentCard.card.getType()) {
                    case CardType::ATTACK: useAttackCard(); break;
                    case CardType::SKILL: useSkillCard(); break;
                    case CardType::POWER: usePowerCard(); break;
                    case CardType::STATUS: case CardType::CURSE: onUseStatusOrCurseCard(); break;
                    default: break;
                }
                break;
            case P::CardAfterChoice:
                resolution[index].phase = P::CardTriggers;
                if (resolution[index].drawsAfterChoice) Actions::DrawCards(*this, resolution[index].drawsAfterChoice);
                break;
            case P::CardTriggers:
                resolution[index].phase = P::CardCleanup;
                if (outcome != Outcome::UNDECIDED) break;
                switch (currentCard.card.getType()) {
                    case CardType::ATTACK: onUseAttackCard(); break;
                    case CardType::SKILL: onUseSkillCard(); break;
                    case CardType::POWER: onUsePowerCard(); break;
                    default: break;
                }
                break;
            case P::CardCleanup:
                resolution[index].phase = P::CardDone;
                onAfterUseCard();
                break;
            case P::EndRelics:
                resolution[index].phase = P::EndPowers;
                callEndOfTurnActions();
                break;
            case P::EndPowers:
                resolution[index].phase = P::EndDiscard;
                onTurnEnding();
                break;
            case P::EndDiscard:
                resolution[index].phase = P::Monsters;
                discardAtEndOfTurn();
                turnHasEnded = true;
                monsterTurnIdx = 0;
                if (!skipMonsterTurn) monsters.applyPreTurnLogic(*this);
                break;
            case P::Monsters:
                if (!skipMonsterTurn && monsterTurnIdx < monsters.monsterCount) monsters.doMonsterTurn(*this);
                else { monsters.skipTurn.reset(); resolution[index].phase = P::RoundEnd; }
                break;
            case P::RoundEnd:
                resolution[index].phase = P::StartPowers;
                afterMonsterTurns();
                break;
            case P::StartPowers:
                resolution[index].phase = P::StartDraw;
                player.applyStartOfTurnPowers(*this);
                break;
            case P::StartDraw:
                resolution[index].phase = P::StartDiscards;
                resolution[index].remainingDiscards = player.getStatus<PS::TOOLS_OF_THE_TRADE>();
                finishStartTurn();
                break;
            case P::StartDiscards:
                if (resolution[index].remainingDiscards > 0) {
                    --resolution[index].remainingDiscards;
                    Actions::DiscardOne(*this);
                } else resolution[index].phase = P::Done;
                break;
            case P::CardDone: case P::Done:
                resolution.pop_back();
                continue;
        }
        if (card) resolution[index].card = currentCard;
        checkCombat();
    }
}

bool BattleContext::prepareCard() {
    auto& item = currentCard;
    auto& c = item.card;
    if (item.randomTarget) item.target = monsters.getRandomMonsterIdx(cardRandomRng);
    if (!item.triggerOnUse) { useNoTriggerCard(); return false; }
    const bool usable = item.purgeOnUse || (c.canUse(*this, item.target, item.autoplay) &&
        (!c.requiresTarget() || monsters.arr[item.target].isTargetable()));
    if (!usable) return false;
    if (c.isFreeToPlay(*this)) c.freeToPlayOnce = true;
    if (c.requiresTarget()) player.lastTargetedMonster = item.target;
    item.exhaustOnUse |= c.doesExhaust() || (c.getType() == CardType::SKILL && player.hasStatus<PS::CORRUPTION>());
    ++player.cardsPlayedThisTurn;
    triggerOnOtherCardPlayed(c);
    if (!item.purgeOnUse) {
        cards.removeFromHandById(c.uniqueId);
        if (c.costForTurn > 0 && !c.isFreeToPlay(*this) && !item.autoplay &&
            !(player.hasStatus<PS::CORRUPTION>() && c.getType() == CardType::SKILL)) player.useEnergy(c.costForTurn);
    }
    return true;
}

void BattleContext::drawAfterChoice(int count) {
    if (inputState == InputState::CARD_SELECT) resolution.back().drawsAfterChoice = count;
    else drawCards(count);
}

void BattleContext::useNoTriggerCard() {
    auto &item = currentCard;
    auto &c = item.card;

    switch (c.id) {
        case CardId::BURN:
            Actions::DamagePlayer(*this, c.isUpgraded() ? 4 : 2, true);
            break;

        case CardId::DECAY:
            Actions::DamagePlayer(*this, 2, true);
            break;

        case CardId::DOUBT:
            player.debuff<PS::WEAK>(1, true);
            break;

        case CardId::REGRET:
            Actions::PlayerLoseHp(*this, item.regretCardCount, true);
            break;

        case CardId::SHAME:
            // todo this is fixed just test -> this and doubt are bugged if you are already weak i think. because the apply power action does not create a new power and justapplied is not set to true
            player.debuff<PS::FRAIL>(1, true);
            break;


        default:
            // this can actually be called on any card now because of time warp power

//#ifdef sts_asserts
//            assert(false);
//#endif // sts_asserts
            break;
    }

    cards.removeFromHandById(c.uniqueId);
    Actions::DiscardNoTriggerCard(*this); // todo what if havoc plays one of these
}

void BattleContext::useAttackCard() {
    auto &item = currentCard;
    auto &c = item.card;

    const auto t = item.target;
    const bool up = c.isUpgraded();

    // todo test vigor with multi attacks and necro/double tap
    switch (c.getId()) {
        case CardId::STRIKE_RED:
        case CardId::STRIKE_BLUE:
        case CardId::STRIKE_GREEN:
        case CardId::STRIKE_PURPLE: {
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 9 : 6));
            break;
        }

        case CardId::ANGER:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 8 : 6));
            Actions::MakeTempCardInDiscard(*this,  CardInstance(CardId::ANGER, up), 1);
            break;

        case CardId::ALL_FOR_ONE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 14 : 10));
            Actions::AllForOneAction(*this);
            break;

        case CardId::ALL_OUT_ATTACK:
            Actions::AttackAllEnemy(*this, (up ? 14 : 10) + player.getStatus<PS::VIGOR>());
            Actions::RandomDiscardAction(*this);
            break;

        case CardId::BACKSTAB:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 15 : 11));
            break;

        case CardId::BASH:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 8));
            Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, up ? 3 : 2, false);
            break;

        case CardId::BOWLING_BASH: {
            int count = 0;
            for (int i = 0; i < monsters.monsterCount; ++i) count += !monsters.arr[i].isDeadOrEscaped();
            const int damage = calculateCardDamage(c, t, up ? 8 : 7);
            for (int i = 0; i < count; ++i) Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::CONSECRATE:
            Actions::AttackAllEnemy(*this, (up ? 8 : 5) + player.getStatus<PS::VIGOR>());
            break;

        case CardId::CONCLUDE:
            Actions::AttackAllEnemy(*this, (up ? 16 : 12) + player.getStatus<PS::VIGOR>());
            endTurnAfterCurrentCard = true;
            break;

        case CardId::BANE: {
            const int baseDamage = up ? 10 : 7;
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, baseDamage));
            if (monsters.arr[t].hasStatus<MS::POISON>()) {
                Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, baseDamage));
            }
            break;
        }

        case CardId::MASTERFUL_STAB:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 16 : 12));
            break;

        case CardId::BARRAGE: {
            const int damage = calculateCardDamage(c, t, up ? 6 : 4);
            for (int i = 0; i < player.orbSlots; ++i) {
                if (player.orbs[i] != Orb::EMPTY) {
                    Actions::AttackEnemy(*this, t, damage);
                }
            }
            break;
        }

        case CardId::BEAM_CELL:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, up ? 3 : 2, false);
            break;

        case CardId::BULLSEYE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 11 : 8));
            Actions::DebuffEnemy<MS::LOCK_ON>(*this, t, up ? 3 : 2, false);
            break;

        case CardId::BALL_LIGHTNING:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            Actions::ChannelOrb(*this, Orb::LIGHTNING);
            break;

        case CardId::BITE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 8 : 7));
            Actions::HealPlayer(*this, up ? 3 : 2);
            break;

        case CardId::BODY_SLAM:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, player.block));
            break;

        case CardId::BLOOD_FOR_BLOOD:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 22 : 18));
            break;

        case CardId::BLUDGEON:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 42 : 32));
            break;

        case CardId::BLIZZARD:
            Actions::GainBlock(*this, calculateCardBlock((up ? 3 : 2) * player.frostOrbsChanneled));
            break;

        case CardId::CARNAGE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 28 : 20));
            break;

        case CardId::CHOKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 15 : 12));
            Actions::BuffEnemy<MS::CHOKED>(*this, t, up ? 5 : 3);
            break;

        case CardId::CLASH:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 18 : 14));
            break;

        case CardId::CLEAVE:  {
            int baseDamage = (up ? 11 : 8) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, baseDamage);
            break;
        }

        case CardId::CLOTHESLINE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 14 : 12));
            Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 3 : 2, false);
            break;

        case CardId::CUT_THROUGH_FATE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            Actions::DrawCards(*this, 1);
            break;

        case CardId::CRUSH_JOINTS:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 8));
            if (player.skillsPlayedThisTurn > 0) Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, up ? 2 : 1, false);
            break;

        case CardId::COLD_SNAP:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 9 : 6));
            Actions::ChannelOrb(*this, Orb::FROST);
            break;

        case CardId::CLAW: {
            const int increment = up ? 3 : 2;
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, (up ? 5 : 3) + c.specialData));
            c.specialData += increment;
            if (item.purgeOnUse) {
                cards.findAndUpgradeSpecialData(c.uniqueId, increment);
            }
            break;
        }

        case CardId::COMPILE_DRIVER: {
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            bool seen[5]{};
            int distinct = 0;
            for (int i = 0; i < player.orbSlots; ++i) {
                const int orb = static_cast<int>(player.orbs[i]);
                if (orb != static_cast<int>(Orb::EMPTY) && !seen[orb]) {
                    seen[orb] = true;
                    ++distinct;
                }
            }
            Actions::DrawCards(*this, distinct);
            break;
        }

        case CardId::CORE_SURGE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 15 : 11));
            Actions::BuffPlayer<PS::ARTIFACT>(*this, 1);
            break;

        case CardId::DAGGER_THROW:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 9));
            Actions::DrawCards(*this, 1);
            Actions::DiscardOne(*this);
            break;

        case CardId::DOOM_AND_GLOOM: {
            const int damage = (up ? 10 : 7) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, damage);
            Actions::ChannelOrb(*this, Orb::DARK);
            break;
        }

        case CardId::DRAMATIC_ENTRANCE: {
            int baseDamage = (up ? 12 : 8) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, baseDamage);
            break;
        }

        case CardId::DIE_DIE_DIE: {
            const int baseDamage = (up ? 17 : 13) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, baseDamage);
            break;
        }

        case CardId::DAGGER_SPRAY: {
            const int baseDamage = (up ? 6 : 4) + player.getStatus<PS::VIGOR>();
            for (int i = 0; i < 2; ++i) {
                Actions::AttackAllEnemy(*this, baseDamage);
            }
            break;
        }

        case CardId::DASH:
            Actions::GainBlock(*this, calculateCardBlock(up ? 13 : 10));
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 13 : 10));
            break;

        case CardId::ENDLESS_AGONY:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 6 : 4));
            break;

        case CardId::EMPTY_FIST:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 14 : 9));
            Actions::ChangeStance(*this, Stance::NEUTRAL);
            break;

        case CardId::EXPUNGER:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 15 : 9));
            break;

        case CardId::FLYING_SLEEVES: {
            const int damage = calculateCardDamage(c, t, up ? 6 : 4);
            Actions::AttackEnemy(*this, t, damage);
            Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::FOLLOW_UP:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 11 : 7));
            if (player.cardsPlayedThisTurn > 1) Actions::GainEnergy(*this, 1);
            break;

        case CardId::JUST_LUCKY:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 4 : 3));
            break;

        case CardId::SASH_WHIP: {
            const int damage = calculateCardDamage(c, t, up ? 10 : 8);
            Actions::AttackEnemy(*this, t, damage); Actions::AttackEnemy(*this, t, damage);
            if (player.attacksPlayedThisTurn > 1) Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 2 : 1, false);
            break;
        }

        case CardId::SANDS_OF_TIME:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 26 : 20));
            break;

        case CardId::SIGNATURE_MOVE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 40 : 30));
            break;

        case CardId::TALK_TO_THE_HAND:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 6 : 5));
            Actions::BuffEnemy<MS::BLOCK_RETURN>(*this, t, up ? 3 : 2);
            break;

        case CardId::TANTRUM: {
            const int damage = calculateCardDamage(c, t, up ? 4 : 3);
            for (int i = 0; i < 3; ++i) Actions::AttackEnemy(*this, t, damage);
            Actions::ChangeStance(*this, Stance::WRATH);
            break;
        }

        case CardId::WINDMILL_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            break;

        case CardId::WALLOP:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 9));
            Actions::GainBlock(*this, calculateCardBlock(up ? 12 : 9));
            break;

        case CardId::DROPKICK:
            Actions::DropkickAction(*this, t);
            break;

        case CardId::FEED:
            Actions::FeedAction(*this, t, calculateCardDamage(c, t, up ? 12 : 10), up);
            break;

        case CardId::CARVE_REALITY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 10 : 6));
            Actions::MakeTempCardInHand(*this, CardId::SMITE);
            break;

        case CardId::LESSON_LEARNED:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 13 : 10));
            break;

        case CardId::REACH_HEAVEN:
            Actions::GainBlock(*this, calculateCardBlock(up ? 13 : 10));
            Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::THROUGH_VIOLENCE), 1, true);
            break;

        case CardId::SMITE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 14 : 12));
            break;

        case CardId::FEAR_NO_EVIL:
            noteEnemyIntentAccess();
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 11 : 8));
            if (monsters.arr[t].isAttacking()) Actions::ChangeStance(*this, Stance::CALM);
            break;

        case CardId::FLURRY_OF_BLOWS: {
            const int damage = calculateCardDamage(c, t, up ? 6 : 4);
            Actions::AttackEnemy(*this, t, damage);
            Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::FIEND_FIRE:
            Actions::FiendFireAction(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            break;

        case CardId::FLYING_KNEE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 11 : 8));
            Actions::BuffPlayer<PS::ENERGIZED>(*this, 1);
            break;

        case CardId::HEEL_HOOK:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 8 : 5));
            if (monsters.arr[t].hasStatus<MS::WEAK>()) {
                Actions::DrawCards(*this, 1);
                Actions::GainEnergy(*this, 1);
            }
            break;

        case CardId::FLASH_OF_STEEL:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 6 : 3));
            Actions::DrawCards(*this, 1);
            break;

        case CardId::HAND_OF_GREED:
            Actions::HandOfGreedAction(*this, t, calculateCardDamage(c, t, up ? 25 : 20), up);
            break;

        case CardId::HEADBUTT:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 9));
            Actions::HeadbuttAction(*this);
            break;

        case CardId::HEAVY_BLADE: {
            const int dmg1 = 14 + ((up ? 4 : 2) * player.getStatus<PS::STRENGTH>());
            const int dmg2 = calculateCardDamage(c, t, dmg1);
            Actions::AttackEnemy(*this, t, dmg2);
            break;
        }

        case CardId::HEMOKINESIS:
            //  attack enemy should recalculate damage, because we can lose hp and therefore gain strength before the dmg, need to test
           // actually no i dont think that is true ^^
            Actions::PlayerLoseHp(*this, 2, true);
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 20 : 15));
            break;

        case CardId::IMMOLATE: {
            int baseDamage = (up ? 28 : 21) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, baseDamage);
            Actions::MakeTempCardInDiscard(*this, {CardId::BURN}, 1);
            break;
        }

        case CardId::HYPERBEAM: {
            const int damage = (up ? 34 : 26) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, damage);
            Actions::DebuffPlayer<PS::FOCUS>(*this, -3, false);
            break;
        }

        case CardId::FINISHER: {
            const int damage = calculateCardDamage(c, t, up ? 8 : 6);
            for (int i = 0; i < player.attacksPlayedThisTurn; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::FLECHETTES: {
            int skillCount = 0;
            for (int i = 0; i < cards.cardsInHand; ++i) {
                skillCount += cards.hand[i].getType() == CardType::SKILL;
            }
            const int damage = calculateCardDamage(c, t, up ? 4 : 3);
            for (int i = 0; i < skillCount; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::GLASS_KNIFE: {
            const int damage = calculateCardDamage(c, t, up ? 12 : 8);
            Actions::AttackEnemy(*this, t, damage);
            Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::FTL:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 4 : 3));
            if (player.cardsPlayedThisTurn <= 3) {
                Actions::DrawCards(*this, 1);
            }
            break;

        case CardId::GO_FOR_THE_EYES:
            noteEnemyIntentAccess();
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 4 : 3));
            if (monsters.arr[t].isAttacking()) {
                Actions::DebuffEnemy<MS::WEAK>(*this, t, 1, false);
            }
            break;

        case CardId::ERUPTION:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 9));
            Actions::ChangeStance(*this, Stance::WRATH);
            break;

        case CardId::EVISCERATE:
            for (int i = 0; i < 3; ++i) {
                Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 9 : 7));
            }
            break;

        case CardId::IRON_WAVE: {
            Actions::GainBlock(*this, calculateCardBlock(up ? 7 : 5));
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 7 : 5));
            break;
        }

        case CardId::MIND_BLAST: {
            const auto damage = calculateCardDamage(c, t, static_cast<int>(cards.drawPile.size()));
            Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::MELTER: {
            const int multiplier = monsters.arr[t].block > 0 ? 3 : 1;
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, (up ? 14 : 10) * multiplier));
            break;
        }

        case CardId::METEOR_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 28 : 24));
            for (int i = 0; i < 3; ++i) {
                Actions::ChannelOrb(*this, Orb::FUSION);
            }
            break;

        case CardId::NEUTRALIZE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 4 : 3));
            Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 2 : 1, false);
            break;

        case CardId::PERFECTED_STRIKE: {
            // hack because we calculate strikeCount while non purge cards are still in hand.
            const int strikeDmg = cards.strikeCount * (up ? 3 : 2);
            const int baseDamage = 6 + strikeDmg;
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, baseDamage));
            break;
        }

        case CardId::POISONED_STAB:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 8 : 6));
            Actions::DebuffEnemy<MS::POISON>(*this, t, up ? 4 : 3);
            break;

        case CardId::POMMEL_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 9));
            Actions::DrawCards(*this, up  ? 2 : 1);
            break;

        case CardId::PUMMEL: {
            const auto attackCount = up ? 5 : 4;
            const auto damage = calculateCardDamage(c, t, 2);
            for (int i = 0; i < attackCount; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::QUICK_SLASH:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 8));
            Actions::DrawCards(*this, 1);
            break;

        case CardId::RIP_AND_TEAR: {
            const int damage = calculateCardDamage(c, t, up ? 9 : 7);
            Actions::AttackEnemy(*this, t, damage);
            Actions::AttackEnemy(*this, t, damage);
            break;
        }

        case CardId::RIDDLE_WITH_HOLES: {
            const int damage = calculateCardDamage(c, t, up ? 4 : 3);
            for (int i = 0; i < 5; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::SKEWER: {
            const int energy = std::min(item.energyOnUse, player.energy);
            player.useEnergy(energy);
            const int damage = calculateCardDamage(c, t, up ? 8 : 7);
            for (int i = 0; i < energy; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::RAMPAGE: { // todo doubled card damage is changed by the first
            const int damage = calculateCardDamage(c, t, 8+c.specialData);
            Actions::AttackEnemy(*this, t, damage);

            if (item.purgeOnUse) {
                cards.findAndUpgradeSpecialData(c.uniqueId, up ? 8 : 5);
            }
            c.specialData += up ? 8 : 5;

            break;
        }

        case CardId::REAPER: {
            const auto baseDamage = (up ? 5 : 4) + player.getStatus<PS::VIGOR>();
            Actions::ReaperAction(*this, baseDamage);
            break;
        }

        case CardId::RECKLESS_CHARGE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            Actions::MakeTempCardInDrawPile(*this, {CardId::DAZED}, 1, true);
            break;

        case CardId::RITUAL_DAGGER:
            Actions::RitualDaggerAction(*this, t, calculateCardDamage(c, t, c.specialData));
            break;

        case CardId::RAGNAROK: {
            const int damage = calculateCardDamage(c, t, up ? 6 : 5);
            for (int i = 0; i < (up ? 6 : 5); ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::GRAND_FINALE:
            Actions::AttackAllEnemy(*this, (up ? 60 : 50) + player.getStatus<PS::VIGOR>());
            break;

        case CardId::SEARING_BLOW: {
            const int n = c.getUpgradeCount();
            const int baseDmg = n * (n+7) / 2 + 12;
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, baseDmg));
            break;
        }

        case CardId::SCRAPE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            Actions::ScrapeAction(*this, up ? 5 : 4);
            break;

        case CardId::SEVER_SOUL: // another example of damage being calculated after an action, maybe need a new action to calculate damage later...
            Actions::SeverSoulExhaustAction(*this);
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 22 : 16));
            break;

        case CardId::SHIV:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 6 : 4));
            break;

        case CardId::SLICE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 9 : 6));
            break;

        case CardId::SWIFT_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 10 : 7));
            break;

        case CardId::WEAVE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 6 : 4));
            break;

        case CardId::WHEEL_KICK:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 20 : 15));
            Actions::DrawCards(*this, 2);
            break;

        case CardId::SNEAKY_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 16 : 12));
            break;

        case CardId::STREAMLINE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 20 : 15));
            Actions::ModifyCurrentCardCost(*this, -1);
            break;

        case CardId::PREDATOR:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 20 : 15));
            Actions::BuffPlayer<PS::DRAW_CARD_NEXT_TURN>(*this, up ? 3 : 2);
            break;

        case CardId::UNLOAD:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 18 : 14));
            for (int i = cards.cardsInHand - 1; i >= 0; --i) {
                const auto card = cards.hand[i];
                if (card.getType() != CardType::ATTACK) {
                    cards.removeFromHandAtIdx(i);
                    cards.moveToDiscardPile(card);
                    onManualDiscard(card);
                }
            }
            break;

        case CardId::SUNDER:
            Actions::SunderAction(*this, t, calculateCardDamage(c, t, up ? 32 : 24));
            break;

        case CardId::SWEEPING_BEAM: {
            const int damage = (up ? 9 : 6) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, damage);
            Actions::DrawCards(*this, 1);
            break;
        }

        case CardId::REBOUND:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 12 : 9));
            // The end-of-card hook advances this from 2 to 1; the next
            // playable card then returns to the top of the draw pile.
            Actions::BuffPlayer<PS::REBOUND>(*this, 2);
            break;

        case CardId::SUCKER_PUNCH:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 9 : 7));
            Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 2 : 1, false);
            break;

        case CardId::SWORD_BOOMERANG:
            for (int i = 0; i < (up ? 4 : 3); ++i) {
                Actions::SwordBoomerangAction(*this, 3+player.getStatus<PS::VIGOR>()); // vigor is removed afterwards so this is a necessary (maybe not 100% accurate) hack
            }
            break;

        case CardId::THUNDERCLAP: {
            int baseDamage = (up ? 7 : 4) + player.getStatus<PS::VIGOR>();
            Actions::AttackAllEnemy(*this, baseDamage);
            Actions::DebuffAllEnemy<MS::VULNERABLE>(*this, 1, false);
            break;
        }

        case CardId::THROUGH_VIOLENCE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 30 : 20));
            break;

        case CardId::THUNDER_STRIKE: {
            const int damage = calculateCardDamage(c, t, up ? 9 : 7);
            for (int i = 0; i < player.lightningOrbsChanneled; ++i) {
                Actions::AttackEnemy(*this, t, damage);
            }
            break;
        }

        case CardId::TWIN_STRIKE: { // todo test with breaking block on first hit and drill relic
            const int dmg = calculateCardDamage(c, t, up ? 7 : 5);
            Actions::AttackEnemy(*this, t, dmg);
            Actions::AttackEnemy(*this, t, dmg);
            break;
        }

        case CardId::UPPERCUT:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, 13));
            Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 2 : 1, false);
            Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, up ? 2 : 1, false);
            break;

        case CardId::WHIRLWIND: {
            if (!item.ignoreEnergyTotal && player.energy < item.energyOnUse) {
                item.energyOnUse = player.energy;
            }
            const auto baseDamage = (up ? 8 : 5) + player.getStatus<PS::VIGOR>();
            Actions::WhirlwindAction(*this, baseDamage, item.energyOnUse, !(item.freeToPlay || c.freeToPlayOnce));
            break;
        }

        case CardId::WILD_STRIKE:
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t, up ? 17 : 12));
            Actions::MakeTempCardInDrawPile(*this,  CardInstance(CardId::WOUND), 1, true);
            break;

        case CardId::BRILLIANCE:
            // Brilliance reads the current Mantra total when played; it does not
            // permanently modify the card's base damage.
            Actions::AttackEnemy(*this, t, calculateCardDamage(c, t,
                (up ? 16 : 12) + player.getStatus<PS::MANTRA>()));
            break;


        default:
            unsupportedEffectKind = UnsupportedEffectKind::CARD;
            unsupportedEffectId = static_cast<int>(c.id);
#ifdef sts_asserts
            std::cerr << "attempted to use unimplemented card: " << c.getName() << std::endl;
            assert(false);
#endif
            break;
    }
}

// todo a lot of things can be done immediately
void BattleContext::useSkillCard() {
    auto &item = currentCard;
    auto &c = item.card;
    const auto t = item.target;
    const bool up = c.isUpgraded();

    switch (c.getId()) {
        case CardId::DEFEND_RED:
        case CardId::DEFEND_BLUE:
        case CardId::DEFEND_GREEN:
        case CardId::DEFEND_PURPLE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 8 : 5));
            break;

        case CardId::ARMAMENTS:
            Actions::GainBlock(*this, calculateCardBlock(5));
            if (up) {
                Actions::UpgradeAllCardsInHand(*this);
            } else {
                Actions::ArmamentsAction(*this);
            }
            break;

        case CardId::APOTHEOSIS:
            Actions::ApotheosisAction(*this);
            break;

        case CardId::AMPLIFY:
            Actions::BuffPlayer<PS::AMPLIFY>(*this, up ? 2 : 1);
            break;

        case CardId::AGGREGATE:
            Actions::GainEnergy(*this, static_cast<int>(cards.drawPile.size()) / (up ? 3 : 4));
            break;

        case CardId::AUTO_SHIELDS:
            if (player.block == 0) {
                Actions::GainBlock(*this, calculateCardBlock(up ? 15 : 11));
            }
            break;

        case CardId::APPARITION:
            Actions::BuffPlayer<PS::INTANGIBLE>(*this, 1);
            break;

        case CardId::BANDAGE_UP:
            Actions::HealPlayer(*this, up ? 6 : 4);
            break;

        case CardId::BATTLE_TRANCE:
            Actions::DrawCards(*this, up ? 4 : 3);
            Actions::DebuffPlayer<PS::NO_DRAW>(*this);
            break;

        case CardId::CRESCENDO:
            Actions::ChangeStance(*this, Stance::WRATH);
            break;

        case CardId::ALPHA:
            Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::BETA), 1, true);
            break;

        case CardId::BETA:
            Actions::BuffPlayer<PS::OMEGA>(*this, 50);
            break;

        case CardId::BLASPHEMY:
            Actions::ChangeStance(*this, Stance::DIVINITY);
            Actions::BuffPlayer<PS::BLASPHEMER>(*this, 1);
            break;

        case CardId::CONJURE_BLADE: {
            const int energy = std::min(item.energyOnUse, player.energy);
            player.useEnergy(energy);
            for (int i = 0; i < energy; ++i) Actions::MakeTempCardInHand(*this, CardId::EXPUNGER, up);
            break;
        }

        case CardId::FOREIGN_INFLUENCE:
            Actions::MakeRandomCardInHand(*this, CardType::ATTACK, 1, true);
            break;

        case CardId::INDIGNATION:
            Actions::BuffPlayer<PS::MANTRA>(*this, up ? 3 : 2);
            break;

        case CardId::PRESSURE_POINTS:
            Actions::BuffEnemy<MS::MARK>(*this, t, up ? 11 : 8);
            break;

        case CardId::JUDGMENT:
            {
                auto &monster = monsters.arr[t];
                if (!monster.isDeadOrEscaped() && monster.curHp <= (up ? 40 : 30)) {
                    monster.curHp = 0;
                    monster.die(*this);
                    checkCombat();
                }
            }
            break;

        case CardId::CARVE_REALITY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 10 : 6));
            Actions::MakeTempCardInHand(*this, CardId::SMITE);
            break;

        case CardId::COLLECT: {
            const int energy = std::min(item.energyOnUse, player.energy);
            player.useEnergy(energy);
            Actions::MakeTempCardInHand(*this, CardId::MIRACLE, false, energy);
            break;
        }

        case CardId::EMPTY_MIND:
            Actions::ChangeStance(*this, Stance::CALM);
            Actions::DrawCards(*this, 2);
            break;

        case CardId::EVALUATE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 10 : 6));
            Actions::MakeTempCardInHand(*this, CardId::INSIGHT);
            break;

        case CardId::DEUS_EX_MACHINA:
            Actions::MakeTempCardInHand(*this, CardId::MIRACLE, false, up ? 3 : 2);
            break;

        case CardId::INSIGHT:
            Actions::DrawCards(*this, 2);
            break;

        case CardId::EMPTY_BODY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 11 : 7));
            Actions::ChangeStance(*this, Stance::NEUTRAL);
            break;

        case CardId::HALT:
            Actions::GainBlock(*this, calculateCardBlock(player.stance == Stance::CALM ? (up ? 18 : 14) : (up ? 12 : 9)));
            break;

        case CardId::INNER_PEACE:
            if (player.stance == Stance::CALM) Actions::DrawCards(*this, up ? 4 : 3);
            else Actions::ChangeStance(*this, Stance::CALM);
            break;

        case CardId::MEDITATE:
            Actions::ChangeStance(*this, Stance::CALM);
            break;

        case CardId::OMNISCIENCE:
            if (!cards.drawPile.empty()) {
                openSimpleCardSelectScreen(CardSelectTask::OMNISCIENCE, 1);
            }
            break;

        case CardId::PERSEVERANCE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 12 : 7));
            break;

        case CardId::PRAY:
            Actions::BuffPlayer<PS::MANTRA>(*this, up ? 4 : 3);
            Actions::MakeTempCardInHand(*this, CardId::MIRACLE);
            break;

        case CardId::REACH_HEAVEN:
            Actions::GainBlock(*this, calculateCardBlock(up ? 13 : 10));
            Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::THROUGH_VIOLENCE), 1, true);
            break;

        case CardId::SIMMERING_FURY:
            Actions::ChangeStance(*this, Stance::WRATH);
            Actions::DrawCards(*this, up ? 3 : 2);
            break;

        case CardId::SWIVEL:
            Actions::BuffPlayer<PS::VIGOR>(*this, up ? 10 : 8);
            break;

        case CardId::PROTECT:
            Actions::GainBlock(*this, calculateCardBlock(up ? 16 : 12));
            break;

        case CardId::PROSTRATE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 6 : 4));
            Actions::BuffPlayer<PS::MANTRA>(*this, up ? 3 : 2);
            break;

        case CardId::SANCTITY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 9 : 6));
            if (player.stance == Stance::CALM) Actions::DrawCards(*this, 2);
            break;

        case CardId::SAFETY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 16 : 12));
            break;

        case CardId::SPIRIT_SHIELD:
            Actions::GainBlock(*this, calculateCardBlock((up ? 4 : 3) * cards.cardsInHand));
            break;

        case CardId::THIRD_EYE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 9 : 7));
            break;

        case CardId::TRANQUILITY:
            Actions::ChangeStance(*this, Stance::CALM);
            break;

        case CardId::WORSHIP:
            Actions::BuffPlayer<PS::MANTRA>(*this, up ? 6 : 5);
            break;

        case CardId::VAULT:
            skipMonsterTurn = true;
            endTurnAfterCurrentCard = true;
            break;

        case CardId::WAVE_OF_THE_HAND:
            Actions::BuffPlayer<PS::WAVE_OF_THE_HAND>(*this, up ? 2 : 1);
            break;

        case CardId::WREATH_OF_FLAME:
            Actions::BuffPlayer<PS::VIGOR>(*this, up ? 8 : 5);
            break;

        case CardId::WISH:
            openSimpleCardSelectScreen(CardSelectTask::WISH, 1);
            break;

        case CardId::BULLET_TIME:
            Actions::BulletTimeAction(*this);
            break;

        case CardId::DOPPELGANGER: {
            const int energy = std::min(item.energyOnUse, player.energy);
            player.useEnergy(energy);
            Actions::BuffPlayer<PS::ENERGIZED>(*this, energy);
            Actions::BuffPlayer<PS::DRAW_CARD_NEXT_TURN>(*this, energy);
            break;
        }

        case CardId::BLIZZARD:
            Actions::GainBlock(*this, calculateCardBlock((up ? 3 : 2) * player.frostOrbsChanneled));
            break;

        case CardId::BOOT_SEQUENCE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 13 : 10));
            break;

        case CardId::ADRENALINE:
            Actions::DrawCards(*this, up ? 3 : 2);
            Actions::GainEnergy(*this, 1);
            break;

        case CardId::ALCHEMIZE:
            obtainPotion(returnRandomPotion(potionRng, player.cc, true));
            break;

        case CardId::OUTMANEUVER:
            Actions::BuffPlayer<PS::ENERGIZED>(*this, up ? 4 : 3);
            break;

        case CardId::PIERCING_WAIL:
            Actions::DebuffAllEnemy<MS::STRENGTH>(*this, -(up ? 8 : 6), false);
            break;

        case CardId::ACROBATICS:
            Actions::DrawCards(*this, up ? 4 : 3);
            Actions::DiscardOne(*this);
            break;

        case CardId::BACKFLIP:
            Actions::GainBlock(*this, calculateCardBlock(up ? 8 : 5));
            Actions::DrawCards(*this, 2);
            break;

        case CardId::BOUNCING_FLASK:
            Actions::BouncingFlaskAction(*this, up ? 4 : 3);
            break;

        case CardId::CALCULATED_GAMBLE:
            Actions::CalculatedGambleAction(*this);
            break;

        case CardId::BLADE_DANCE:
            Actions::MakeTempCardInHand(*this, CardId::SHIV, false, up ? 4 : 3);
            break;

        case CardId::BLUR:
            Actions::GainBlock(*this, calculateCardBlock(up ? 8 : 5));
            Actions::BuffPlayer<PS::BLUR>(*this);
            break;

        case CardId::BLIND:
            if (up) {
                Actions::DebuffAllEnemy<MS::WEAK>(*this, 2, false);
            } else {
                Actions::DebuffEnemy<MS::WEAK>(*this, t, 2, false);
            }
            break;

        case CardId::BLOODLETTING:
            Actions::PlayerLoseHp(*this, 3, true);
            Actions::GainEnergy(*this, up ? 3 : 2);
            break;

        case CardId::BURNING_PACT:
            Actions::ChooseExhaustOne(*this);
            drawAfterChoice(up ? 3 : 2);
            break;

        case CardId::CHILL:
            for (int i = 0; i < (up ? 2 : 1); ++i) {
                Actions::ChannelOrb(*this, Orb::FROST);
            }
            break;

        case CardId::CONSUME:
            if (player.orbSlots > 0) {
                --player.orbSlots;
                player.orbs[player.orbSlots] = Orb::EMPTY;
                player.orbData[player.orbSlots] = 0;
            }
            Actions::BuffPlayer<PS::FOCUS>(*this, up ? 3 : 2);
            break;

        case CardId::CONCENTRATE:
            for (int i = cards.cardsInHand - 1, initialHand = cards.cardsInHand,
                     discardCount = std::min(initialHand, up ? 2 : 3);
                 i >= initialHand - discardCount; --i) {
                const auto card = cards.hand[i];
                cards.removeFromHandAtIdx(i);
                cards.moveToDiscardPile(card);
                onManualDiscard(card);
            }
            Actions::GainEnergy(*this, up ? 3 : 2);
            break;

        case CardId::CHARGE_BATTERY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 10 : 7));
            Actions::BuffPlayer<PS::ENERGIZED>(*this, 1);
            break;

        case CardId::CHRYSALIS:
            Actions::PutRandomCardsInDrawPile(*this, CardType::SKILL, up ? 5 : 3);
            break;

        case CardId::DARK_SHACKLES:
            Actions::DebuffEnemy<MS::STRENGTH>(*this, t, up ? 15 : 9);
            if (monsters.arr[t].hasStatus<MS::ARTIFACT>()) {
                Actions::BuffEnemy<MS::SHACKLED>(*this, t, up ? 15 : 9);
            }
            break;

        case CardId::DARKNESS:
            Actions::ChannelOrb(*this, Orb::DARK);
            break;

        case CardId::CLOAK_AND_DAGGER:
            Actions::GainBlock(*this, calculateCardBlock(up ? 8 : 6));
            Actions::MakeTempCardInHand(*this, CardId::SHIV, false, up ? 2 : 1);
            break;

        case CardId::ESCAPE_PLAN:
            Actions::EscapePlanAction(*this, up ? 5 : 3);
            break;

        case CardId::COOLHEADED:
            Actions::ChannelOrb(*this, Orb::FROST);
            Actions::DrawCards(*this, up ? 2 : 1);
            break;

        case CardId::CHAOS:
            for (int i = 0; i < (up ? 2 : 1); ++i) {
                Actions::ChannelOrb(*this, static_cast<Orb>(1 + cardRandomRng.random(3)));
            }
            break;

        case CardId::EQUILIBRIUM:
            Actions::GainBlock(*this, calculateCardBlock(up ? 16 : 13));
            Actions::BuffPlayer<PS::EQUILIBRIUM>(*this, 1);
            break;

        case CardId::DEADLY_POISON:
            Actions::DebuffEnemy<MS::POISON>(*this, t, up ? 7 : 5);
            break;

        case CardId::CRIPPLING_CLOUD:
            Actions::DebuffAllEnemy<MS::WEAK>(*this, up ? 3 : 2, false);
            Actions::DebuffAllEnemy<MS::POISON>(*this, up ? 7 : 4);
            break;

        case CardId::CATALYST:
            {
                auto &monster = monsters.arr[t];
                if (!monster.isDeadOrEscaped() && monster.hasStatus<MS::POISON>()) {
                    monster.setStatus<MS::POISON>(monster.getStatus<MS::POISON>() * (up ? 3 : 2));
                }
            }
            break;

        case CardId::CORPSE_EXPLOSION:
            Actions::DebuffEnemy<MS::POISON>(*this, t, up ? 9 : 6);
            Actions::DebuffEnemy<MS::CORPSE_EXPLOSION>(*this, t, up ? 3 : 2);
            break;

        case CardId::DISTRACTION:
            Actions::MakeRandomCardInHand(*this, CardType::SKILL, 1, true);
            break;

        case CardId::DEFLECT:
            Actions::GainBlock(*this, calculateCardBlock(up ? 7 : 4));
            break;

        case CardId::DEEP_BREATH:
            if (!cards.discardPile.empty()) {
                onShuffle();
                Actions::EmptyDeckShuffle(*this);
                Actions::ShuffleDrawPile(*this);
            }
            Actions::DrawCards(*this, up ? 2 : 1);
            break;

        case CardId::DECEIVE_REALITY:
            Actions::GainBlock(*this, calculateCardBlock(up ? 6 : 4));
            Actions::MakeTempCardInHand(*this, CardId::SAFETY);
            break;

        case CardId::DOUBLE_ENERGY:
            Actions::GainEnergy(*this, player.energy);
            break;

        case CardId::DODGE_AND_ROLL:
            Actions::GainBlock(*this, calculateCardBlock(up ? 6 : 4));
            Actions::BuffPlayer<PS::NEXT_TURN_BLOCK>(*this, up ? 6 : 4);
            break;

        case CardId::DISARM:
            Actions::DebuffEnemy<MS::STRENGTH>(*this, t, up ? -3 : -2, false);
            break;

        case CardId::SURVIVOR:
            Actions::GainBlock(*this, calculateCardBlock(up ? 11 : 8));
            Actions::DiscardOne(*this);
            break;

        case CardId::ZAP:
            Actions::ChannelOrb(*this, Orb::LIGHTNING);
            break;

        case CardId::DISCOVERY:
            undefinedBehaviorEvoked = true;
            Actions::DiscoveryAction(*this, CardType::INVALID, 1);
            break;

        case CardId::DOUBLE_TAP:
            Actions::BuffPlayer<PS::DOUBLE_TAP>(*this, up ? 2 : 1);
            break;

        case CardId::DUAL_WIELD:
            Actions::DualWieldAction(*this, up ? 2 : 1);
            break;

        case CardId::DUALCAST:
            Actions::EvokeFrontOrb(*this);
            Actions::EvokeFrontOrb(*this);
            break;

        case CardId::ENLIGHTENMENT:
            Actions::EnlightenmentAction(*this, up);
            break;

        case CardId::ENTRENCH:
            Actions::EntrenchAction(*this);
            break;

        case CardId::EXHUME:
            Actions::ExhumeAction(*this);
            break;

        case CardId::MALAISE: {
            const int energy = std::min(item.energyOnUse, player.energy);
            player.useEnergy(energy);
            Actions::DebuffEnemy<MS::WEAK>(*this, t, energy, false);
            Actions::DebuffEnemy<MS::STRENGTH>(*this, t, -energy, false);
            break;
        }

        case CardId::NIGHTMARE:
            if (cards.cardsInHand > 0) {
                openSimpleCardSelectScreen(CardSelectTask::NIGHTMARE, 1);
                cardSelectInfo.data0 = up ? 4 : 3;
            }
            break;

        case CardId::BURST:
            Actions::BuffPlayer<PS::BURST>(*this, up ? 2 : 1);
            break;

        case CardId::TACTICIAN:
            // Tactician is normally unplayable; retaining this no-op keeps malformed
            // externally supplied action sequences from corrupting the simulation.
            break;

        case CardId::LEG_SWEEP:
            Actions::GainBlock(*this, calculateCardBlock(up ? 14 : 11));
            Actions::DebuffEnemy<MS::WEAK>(*this, t, up ? 3 : 2, false);
            break;

        case CardId::SETUP:
            Actions::SetupAction(*this);
            break;

        case CardId::TERROR:
            Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, 99, false);
            break;

        case CardId::EXPERTISE:
            Actions::DrawCards(*this, std::max(0, 6 - cards.cardsInHand));
            break;

        case CardId::PHANTASMAL_KILLER:
            Actions::BuffPlayer<PS::PHANTASMAL>(*this);
            break;

        case CardId::FUSION:
            Actions::ChannelOrb(*this, Orb::FUSION, up ? 2 : 1);
            break;

        case CardId::GENETIC_ALGORITHM: {
            Actions::GainBlock(*this, calculateCardBlock((up ? 2 : 1) + c.specialData));
            c.specialData += 1;
            if (item.purgeOnUse) {
                cards.findAndUpgradeSpecialData(c.uniqueId, 1);
            }
            break;
        }

        case CardId::FINESSE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 4 : 2));
            Actions::DrawCards(*this, 1);
            break;

        case CardId::FISSION: {
            int evokeCount = 0;
            for (int i = 0; i < player.orbSlots; ++i) {
                evokeCount += player.orbs[i] != Orb::EMPTY;
            }
            for (int i = 0; i < evokeCount; ++i) {
                Actions::EvokeFrontOrb(*this);
            }
            Actions::GainEnergy(*this, evokeCount);
            if (up) {
                Actions::DrawCards(*this, evokeCount);
            }
            break;
        }

        case CardId::FORCE_FIELD:
            Actions::GainBlock(*this, calculateCardBlock(up ? 16 : 12));
            break;

        case CardId::GLACIER:
            for (int i = 0; i < (up ? 3 : 2); ++i) {
                Actions::ChannelOrb(*this, Orb::FROST);
            }
            break;

        case CardId::FORETHOUGHT:
            Actions::ForethoughtAction(*this, up);
            break;

        case CardId::PREPARED:
            Actions::DrawCards(*this, up ? 2 : 1);
            Actions::DiscardOne(*this);
            break;

        case CardId::FLAME_BARRIER:
            Actions::GainBlock(*this, calculateCardBlock(up ? 16 : 12));
            Actions::BuffPlayer<PS::FLAME_BARRIER>(*this, up ? 6 : 4);
            break;

        case CardId::FLEX:
            Actions::BuffPlayer<PS::STRENGTH>(*this, up ? 4 : 2);
            Actions::DebuffPlayer<PS::LOSE_STRENGTH>(*this, up ? 4 : 2);
            break;

        case CardId::GHOSTLY_ARMOR:
            Actions::GainBlock(*this, calculateCardBlock(up  ?  13 : 10));
            break;

        case CardId::GOOD_INSTINCTS:
            Actions::GainBlock(*this, calculateCardBlock(up  ?  9 : 6));
            break;

        case CardId::HOLOGRAM:
            Actions::GainBlock(*this, calculateCardBlock(up ? 5 : 3));
            Actions::BetterDiscardPileToHandAction(*this, 1, CardSelectTask::HOLOGRAM);
            break;

        case CardId::HAVOC:
            Actions::PlayTopCard(*this, monsters.getRandomMonsterIdx(cardRandomRng, true), true);
            break;

        case CardId::IMPATIENCE: {
            bool hasAttack = false;
            for (int i = 0; i < cards.cardsInHand; ++i) {
                if (cards.hand[i].getType() == CardType::ATTACK) {
                    hasAttack = false;
                    break;
                }
            }
            if (!hasAttack) {
                Actions::DrawCards(*this, up ? 3 : 2);
            }
            break;
        }

        case CardId::IMPERVIOUS:
            Actions::GainBlock(*this, calculateCardBlock(up  ?  40 : 30));
            break;

        case CardId::INFERNAL_BLADE:
            Actions::InfernalBladeAction(*this);
            break;

        case CardId::INTIMIDATE:
            Actions::DebuffAllEnemy<MS::WEAK>(*this, up ? 2 : 1, false); // game justs adds one for each enemy in order
            break;

        case CardId::JACK_OF_ALL_TRADES: // the game decides the random cards here and adds maketempcardtobot
            Actions::JackOfAllTradesAction(*this, up);
            break;

        case CardId::JAX:
            Actions::PlayerLoseHp(*this, 3, true);
            Actions::BuffPlayer<PS::STRENGTH>(*this, up ? 3 : 2);
            break;

        case CardId::LEAP:
            Actions::GainBlock(*this, calculateCardBlock(up ? 12 : 9));
            break;

        case CardId::LIMIT_BREAK:
            Actions::LimitBreakAction(*this);
            break;

        case CardId::MADNESS:
            Actions::MadnessAction(*this);
            break;

        case CardId::MASTER_OF_STRATEGY:
            Actions::DrawCards(*this, up ? 4 : 3);
            break;

        case CardId::METAMORPHOSIS:
            Actions::PutRandomCardsInDrawPile(*this, CardType::ATTACK, up ? 5 : 3);
            break;

        case CardId::MIRACLE:
            Actions::GainEnergy(*this, up ? 2 : 1);
            break;

        case CardId::MULTI_CAST: {
            if (!item.ignoreEnergyTotal && player.energy < item.energyOnUse) {
                item.energyOnUse = player.energy;
            }
            const int evokeCount = item.energyOnUse + (up ? 1 : 0);
            for (int i = 0; i < evokeCount; ++i) {
                Actions::EvokeFrontOrb(*this);
            }
            break;
        }

        case CardId::OVERCLOCK:
            Actions::DrawCards(*this, up ? 3 : 2);
            Actions::MakeTempCardInDiscard(*this, CardInstance(CardId::BURN), 1);
            break;

        case CardId::OFFERING:
            Actions::PlayerLoseHp(*this, 6, true);
            Actions::GainEnergy(*this, 2);
            Actions::DrawCards(*this, up ? 5 : 3);
            break;

        case CardId::PANACEA:
            Actions::BuffPlayer<PS::ARTIFACT>(*this, up ? 2 : 1);
            break;

        case CardId::PANIC_BUTTON:
            Actions::GainBlock(*this, calculateCardBlock(up ? 40 : 30));
            Actions::DebuffPlayer<PS::NO_BLOCK>(*this, 2);
            break;

        case CardId::POWER_THROUGH:
            Actions::MakeTempCardInHand(*this, CardId::WOUND, false, 2);
            Actions::GainBlock(*this, calculateCardBlock(up ? 20 : 15));
            break;

        case CardId::PURITY:
            Actions::ExhaustMany(*this, up ? 5 : 3);
            break;

        case CardId::REPROGRAM:
            Actions::DebuffPlayer<PS::FOCUS>(*this, -1, false);
            Actions::BuffPlayer<PS::STRENGTH>(*this, up ? 2 : 1);
            Actions::BuffPlayer<PS::DEXTERITY>(*this, up ? 2 : 1);
            break;

        case CardId::RAINBOW:
            Actions::ChannelOrb(*this, Orb::LIGHTNING);
            Actions::ChannelOrb(*this, Orb::FROST);
            Actions::ChannelOrb(*this, Orb::DARK);
            break;

        case CardId::RECURSION: {
            const Orb front = player.orbSlots > 0 ? player.orbs[0] : Orb::EMPTY;
            const int frontOrbData = player.orbSlots > 0 ? player.orbData[0] : 0;
            if (front != Orb::EMPTY) {
                Actions::EvokeFrontOrb(*this);
                Actions::ChannelOrb(*this, front, frontOrbData);
            }
            break;
        }

        case CardId::RAGE:
            Actions::BuffPlayer<PS::RAGE>(*this, up ? 5 : 3);
            break;

        case CardId::SECRET_TECHNIQUE:
            Actions::DrawToHandAction(*this, CardSelectTask::SECRET_TECHNIQUE, CardType::SKILL);
            break;

        case CardId::SECRET_WEAPON:
            Actions::DrawToHandAction(*this, CardSelectTask::SECRET_WEAPON, CardType::ATTACK);
            break;

        case CardId::SECOND_WIND:
            Actions::SecondWindAction(*this, calculateCardBlock(up ? 7 : 5));
            break;

        case CardId::SEEING_RED:
            Actions::GainEnergy(*this, 2);
            break;

        case CardId::SENTINEL:
            Actions::GainBlock(*this, calculateCardBlock(up ? 8 : 5));
            break;

        case CardId::SEEK:
            Actions::SeekAction(*this, up ? 2 : 1);
            break;

        case CardId::SHOCKWAVE:
            Actions::DebuffAllEnemy<MS::WEAK>(*this, up ? 5 : 3, false);
            Actions::DebuffAllEnemy<MS::VULNERABLE>(*this, up ? 5 : 3, false);
            break;

        case CardId::REINFORCED_BODY: {
            if (!item.ignoreEnergyTotal && player.energy < item.energyOnUse) {
                item.energyOnUse = player.energy;
            }
            Actions::GainBlock(*this, calculateCardBlock((up ? 9 : 7) * item.energyOnUse));
            break;
        }

        case CardId::REBOOT:
            Actions::RebootAction(*this, up ? 6 : 4);
            break;

        case CardId::RECYCLE:
            Actions::RecycleAction(*this);
            break;

        case CardId::SKIM:
            Actions::DrawCards(*this, up ? 4 : 3);
            break;

        case CardId::STACK: {
            int orbCount = 0;
            for (int i = 0; i < player.orbSlots; ++i) {
                orbCount += player.orbs[i] != Orb::EMPTY;
            }
            Actions::GainBlock(*this, calculateCardBlock((up ? 4 : 3) * orbCount));
            break;
        }

        case CardId::STEAM_BARRIER:
            Actions::GainBlock(*this, calculateCardBlock(std::max(0, (up ? 8 : 6) + c.specialData)));
            --c.specialData;
            break;

        case CardId::STORM_OF_STEEL:
            Actions::StormOfSteelAction(*this, up);
            break;

        case CardId::SHRUG_IT_OFF:
            Actions::GainBlock(*this, calculateCardBlock(up ? 11 : 8));
            Actions::DrawCards(*this, 1);
            break;

        case CardId::SPOT_WEAKNESS:
            Actions::SpotWeaknessAction(*this, t, up ? 4 : 3);
            break;

        case CardId::THE_BOMB:
            Actions::BuffPlayer<PS::THE_BOMB>(*this, up ? 50 : 40);
            break;

        case CardId::THINKING_AHEAD: // same as upgraded warcry
            Actions::DrawCards(*this, 2);
            Actions::WarcryAction(*this);
            break;

        case CardId::TRANSMUTATION:
            if (player.energy > item.energyOnUse) {
                item.energyOnUse = player.energy;
            }
            if (!item.ignoreEnergyTotal && player.energy < item.energyOnUse) {
                item.energyOnUse = player.energy;
            }

            Actions::TransmutationAction(*this, up, item.energyOnUse, !(item.freeToPlay || c.freeToPlayOnce));
            break;

        case CardId::TRIP: // maybe fixed --- todo this doesn't work properly because it only requires a target when not upgraded, also the trip card doesn't uses its own implementation of debuff all enemy
            if (up) {
                Actions::DebuffAllEnemy<MS::VULNERABLE>(*this, 2, false);
            } else {
                Actions::DebuffEnemy<MS::VULNERABLE>(*this, t, 2, false);
            }
            break;

        case CardId::TRUE_GRIT:
            Actions::GainBlock(*this, calculateCardBlock(up ? 9 : 7));
            if (up) {
                Actions::ChooseExhaustOne(*this);
            } else {
                Actions::ExhaustRandomCardInHand(*this, 1);
            }
            break;

        case CardId::VIOLENCE:
            Actions::ViolenceAction(*this, up ? 4 : 3);
            break;

        case CardId::TURBO:
            Actions::GainEnergy(*this, up ? 4 : 3);
            Actions::MakeTempCardInDiscard(*this, CardInstance(CardId::VOID), 1);
            break;

        case CardId::TEMPEST:
            if (!item.ignoreEnergyTotal && player.energy < item.energyOnUse) {
                item.energyOnUse = player.energy;
            }
            for (int i = 0; i < item.energyOnUse; ++i) {
                Actions::ChannelOrb(*this, Orb::LIGHTNING);
            }
            break;

        case CardId::VIGILANCE:
            Actions::GainBlock(*this, calculateCardBlock(up ? 12 : 8));
            Actions::ChangeStance(*this, Stance::CALM);
            break;

        case CardId::WARCRY:
            Actions::DrawCards(*this, up ? 2 : 1);
            Actions::WarcryAction(*this);
            break;

        case CardId::WHITE_NOISE:
            Actions::MakeRandomCardInHand(*this, CardType::POWER, 1, true);
            break;

        default:
            unsupportedEffectKind = UnsupportedEffectKind::CARD;
            unsupportedEffectId = static_cast<int>(c.id);
#ifdef sts_asserts
            std::cerr << "attempted to use unimplemented card: " << c.getName() << std::endl;
            assert(false);
#endif
            break;
    }
}

void BattleContext::usePowerCard() {
    auto &item = currentCard;
    auto &c = item.card;

    const auto t = item.target;
    const bool up = c.isUpgraded();

    switch (c.getId()) {

        case CardId::ACCURACY:
            Actions::BuffPlayer<PS::ACCURACY>(*this, up ? 6 : 4);
            break;

        case CardId::AFTER_IMAGE:
            Actions::BuffPlayer<PS::AFTER_IMAGE>(*this, 1);
            break;

        case CardId::A_THOUSAND_CUTS:
            Actions::BuffPlayer<PS::THOUSAND_CUTS>(*this, up ? 2 : 1);
            break;

        case CardId::BATTLE_HYMN:
            Actions::BuffPlayer<PS::BATTLE_HYMN>(*this, up ? 2 : 1);
            break;

        case CardId::DEVA_FORM:
            Actions::BuffPlayer<PS::DEVA>(*this, up ? 2 : 1);
            break;

        case CardId::DEVOTION:
            Actions::BuffPlayer<PS::DEVOTION>(*this, up ? 3 : 2);
            break;

        case CardId::ESTABLISHMENT:
            Actions::BuffPlayer<PS::ESTABLISHMENT>(*this, up ? 2 : 1);
            break;

        case CardId::LIKE_WATER:
            Actions::BuffPlayer<PS::LIKE_WATER>(*this, up ? 7 : 5);
            break;

        case CardId::MENTAL_FORTRESS:
            player.mentalFortressBlock += up ? 6 : 4;
            break;

        case CardId::MASTER_REALITY:
            Actions::BuffPlayer<PS::MASTER_REALITY>(*this, 1);
            break;

        case CardId::FORESIGHT:
            Actions::BuffPlayer<PS::FORESIGHT>(*this, up ? 4 : 3);
            break;

        case CardId::FASTING:
            Actions::BuffPlayer<PS::STRENGTH>(*this, up ? 4 : 3);
            Actions::BuffPlayer<PS::DEXTERITY>(*this, up ? 4 : 3);
            --player.energyPerTurn;
            break;

        case CardId::NIRVANA:
            player.nirvanaBlock += up ? 4 : 3;
            break;

        case CardId::RUSHDOWN:
            player.rushdownDraw += up ? 3 : 2;
            break;

        case CardId::STUDY:
            player.studyInsights += up ? 2 : 1;
            break;

        case CardId::WAVE_OF_THE_HAND:
            Actions::BuffPlayer<PS::WAVE_OF_THE_HAND>(*this, up ? 2 : 1);
            break;

        case CardId::WREATH_OF_FLAME:
            Actions::BuffPlayer<PS::VIGOR>(*this, up ? 8 : 5);
            break;

        case CardId::WELL_LAID_PLANS:
            // Retain is represented by the existing end-of-turn hand persistence path.
            break;

        case CardId::CALTROPS:
            Actions::BuffPlayer<PS::THORNS>(*this, up ? 5 : 3);
            break;

        case CardId::ENVENOM:
            Actions::BuffPlayer<PS::ENVENOM>(*this, up ? 2 : 1);
            break;

        case CardId::BARRICADE:
            player.setHasStatus<PS::BARRICADE>(true);
            break;

        case CardId::BIASED_COGNITION:
            Actions::BuffPlayer<PS::FOCUS>(*this, up ? 5 : 4);
            Actions::BuffPlayer<PS::BIAS>(*this, 1);
            break;

        case CardId::BUFFER:
            Actions::BuffPlayer<PS::BUFFER>(*this, up ? 2 : 1);
            break;

        case CardId::BERSERK:
            ++player.energyPerTurn;
            Actions::DebuffPlayer<PS::VULNERABLE>(*this, up ? 1 : 2, false);
            break;

        case CardId::BRUTALITY:
            Actions::BuffPlayer<PS::BRUTALITY>(*this, 1);
            break;

        case CardId::CORRUPTION:
            Actions::BuffPlayer<PS::CORRUPTION>(*this);
            break;

        case CardId::CAPACITOR:
            player.increaseOrbSlots(up ? 3 : 2);
            break;

        case CardId::COMBUST:
            Actions::BuffPlayer<PS::COMBUST>(*this, up ? 7 : 5);
            break;

        case CardId::DEMON_FORM:
            Actions::BuffPlayer<PS::DEMON_FORM>(*this, up ? 3 : 2);
            break;

        case CardId::DARK_EMBRACE:
            Actions::BuffPlayer<PS::DARK_EMBRACE>(*this, 1);
            break;

        case CardId::DEFRAGMENT:
            Actions::BuffPlayer<PS::FOCUS>(*this, up ? 2 : 1);
            break;

        case CardId::ECHO_FORM:
            Actions::BuffPlayer<PS::ECHO_FORM>(*this, up ? 2 : 1);
            break;

        case CardId::EVOLVE:
            Actions::BuffPlayer<PS::EVOLVE>(*this, up ? 2 : 1);
            break;

        case CardId::FEEL_NO_PAIN:
            Actions::BuffPlayer<PS::FEEL_NO_PAIN>(*this, up ? 4 : 3);
            break;

        case CardId::FOOTWORK:
            Actions::BuffPlayer<PS::DEXTERITY>(*this, up ? 3 : 2);
            break;

        case CardId::FIRE_BREATHING:
            Actions::BuffPlayer<PS::FIRE_BREATHING>(*this, up ? 10 : 6);
            break;

        case CardId::INFLAME:
            Actions::BuffPlayer<PS::STRENGTH>(*this, up ? 3 : 2);
            break;

        case CardId::JUGGERNAUT:
            Actions::BuffPlayer<PS::JUGGERNAUT>(*this, up ? 7 : 5);
            break;

        case CardId::LOOP:
            Actions::BuffPlayer<PS::LOOP>(*this, up ? 2 : 1);
            break;

        case CardId::MAGNETISM:
            Actions::BuffPlayer<PS::MAGNETISM>(*this, 1);
            break;

        case CardId::MAYHEM:
            Actions::BuffPlayer<PS::MAYHEM>(*this, 1);
            break;

        case CardId::METALLICIZE:
            Actions::BuffPlayer<PS::METALLICIZE>(*this, up ? 4 : 3);
            break;

        case CardId::PANACHE:
            Actions::BuffPlayer<PS::PANACHE>(*this, up ? 14 : 10);
            break;

        case CardId::RUPTURE:
            Actions::BuffPlayer<PS::RUPTURE>(*this, up ? 2 : 1);
            break;

        case CardId::INFINITE_BLADES:
            Actions::BuffPlayer<PS::INFINITE_BLADES>(*this, up ? 2 : 1);
            break;

        case CardId::NOXIOUS_FUMES:
            Actions::BuffPlayer<PS::NOXIOUS_FUMES>(*this, up ? 3 : 2);
            break;

        case CardId::CREATIVE_AI:
            player.creativeAi += up ? 2 : 1;
            break;

        case CardId::ELECTRODYNAMICS:
            player.setHasStatus<PS::ELECTRO>(true);
            break;

        case CardId::HEATSINKS:
            player.heatsinks += up ? 2 : 1;
            break;

        case CardId::HELLO_WORLD:
            player.helloWorld += up ? 2 : 1;
            break;

        case CardId::MACHINE_LEARNING:
            player.machineLearning += up ? 2 : 1;
            break;

        case CardId::STORM:
            player.storm += up ? 2 : 1;
            break;

        case CardId::SELF_REPAIR:
            player.selfRepairHeal += up ? 10 : 7;
            break;

        case CardId::STATIC_DISCHARGE:
            Actions::BuffPlayer<PS::STATIC_DISCHARGE>(*this, up ? 2 : 1);
            break;

        case CardId::TOOLS_OF_THE_TRADE:
            Actions::BuffPlayer<PS::TOOLS_OF_THE_TRADE>(*this);
            break;

        case CardId::SADISTIC_NATURE:
            Actions::BuffPlayer<PS::SADISTIC>(*this, up ? 7 : 5);
            break;

        case CardId::WRAITH_FORM:
            Actions::BuffPlayer<PS::INTANGIBLE>(*this, up ? 3 : 2);
            Actions::DebuffPlayer<PS::WRAITH_FORM>(*this, 1);
            break;

        default:
            unsupportedEffectKind = UnsupportedEffectKind::CARD;
            unsupportedEffectId = static_cast<int>(c.id);
#ifdef sts_asserts
            std::cerr << "attempted to use unimplemented card: " << c.getName() << std::endl;
            assert(false);
#endif
            break;
    }

}


void BattleContext::onUseAttackCard() {
    auto &item = currentCard;
    auto &c = item.card;

    auto &p = player;
    ++p.attacksPlayedThisTurn;

    // ********* Powers onUseCard *********

    if (p.hasStatus<PS::AFTER_IMAGE>()) {
        Actions::GainBlock(*this, p.getStatus<PS::AFTER_IMAGE>());
    }

    if (!item.purgeOnUse && p.hasStatus<PS::DOUBLE_TAP>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::DOUBLE_TAP>();
    }

    if (!item.purgeOnUse && p.hasStatus<PS::DUPLICATION>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::DUPLICATION>();
    }

    const auto echoForm = p.getStatus<PS::ECHO_FORM>();
    if (!item.purgeOnUse && echoForm) {
        const bool echoFormActive = player.cardsPlayedThisTurn - player.echoFormCardsDoubled <= echoForm;
        if (echoFormActive) {
            ++player.echoFormCardsDoubled;
            queuePurgeCard(c, item.target);
        }
    }

    if (p.hasStatus<PS::PANACHE>() && --p.panacheCounter <= 0) {
        Actions::DamageAllEnemy(*this, p.getStatus<PS::PANACHE>());
    }

    if (p.hasStatus<PS::RAGE>()) {
        Actions::GainBlock(*this, p.getStatus<PS::RAGE>());
    }

    if (p.hasStatus<PS::VIGOR>()) {
        p.removeStatus<PS::VIGOR>();
    }

    if (p.hasStatus<PS::FREE_ATTACK_POWER>()) {
        p.decrementStatus<PS::FREE_ATTACK_POWER>();
    }

    if (p.hasStatus<PS::PEN_NIB>()) {
        // todo does this need to be added to bot?
        Actions::RemoveStatus<PS::PEN_NIB>(*this);
    }

    // ********* Relics onUseCard *********
    // todo order of relics

    if (p.hasRelic<R::INK_BOTTLE>()) {
        p.inkBottleCounter++;
        if (p.inkBottleCounter == 10) {
            p.inkBottleCounter = 0;
            Actions::DrawCards(*this, 1);
        }
    }

    if (p.hasRelic<R::KUNAI>() && p.attacksPlayedThisTurn % 3 == 0) {
        Actions::BuffPlayer<PS::DEXTERITY>(*this, 1);
    }

    if (p.hasRelic<R::ORANGE_PELLETS>()) {
        p.orangePelletsCardTypesPlayed.set(static_cast<int>(CardType::ATTACK), true); // set bit 0 true
        if (p.orangePelletsCardTypesPlayed.all()) {
            p.orangePelletsCardTypesPlayed.reset();
            Actions::RemovePlayerDebuffs(*this);
        }
    }

    if (p.hasRelic<R::ORNAMENTAL_FAN>() && p.attacksPlayedThisTurn % 3 == 0) {
        Actions::GainBlock(*this, 4);
    }

    if (p.hasRelic<R::SHURIKEN>() && p.attacksPlayedThisTurn % 3 == 0) {
        Actions::BuffPlayer<PS::STRENGTH>(*this, 1);
    }

    if (p.hasRelic<R::NECRONOMICON>() && !p.haveUsedNecronomiconThisTurn && !item.freeToPlay && !item.purgeOnUse &&
        (c.costForTurn >= 2 || c.isXCost() && item.energyOnUse >= 2)) {
        queuePurgeCard(c, item.target);
        p.haveUsedNecronomiconThisTurn = true;
    }

    if (p.hasRelic<R::PEN_NIB>()) {
        ++p.penNibCounter;
        if (p.penNibCounter == 9) {
            Actions::BuffPlayer<PS::PEN_NIB>(*this, 1);
            p.penNibCounter = -1; // take note of this
        }
    }

    if (p.hasRelic<R::DUALITY>()) {
        Actions::DualityAction(*this);
    }

    if (p.hasRelic<R::NUNCHAKU>()) {
        if (++p.nunchakuCounter >= 10) {
            Actions::GainEnergy(*this, 1);
            p.nunchakuCounter = 0;
        }
    }

    /*
     *  for each card in hand : TriggerOnCardPlayed
     *  for each card in discardPile : TriggerOnCardPlayed
     *  for each card in drawPile : TriggerOnCardPlayed
     */

    // ********* Enemy Powers onUseCard *********

    // todo Choke Power

    auto &m = monsters.arr[0];
    if (m.hasStatus<MS::SHARP_HIDE>()) {
        Actions::DamagePlayer(*this, m.getStatus<MS::SHARP_HIDE>());
    }

}

void BattleContext::onUseSkillCard() {
    auto &item = currentCard;
    auto &c = item.card;

    auto &p = player;
    ++p.skillsPlayedThisTurn;

    // ********* Powers onUseCard *********

    if (p.hasStatus<PS::AFTER_IMAGE>()) {
        Actions::GainBlock(*this, p.getStatus<PS::AFTER_IMAGE>());
    }

    if (!item.purgeOnUse && p.hasStatus<PS::BURST>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::BURST>();
    }

    if (!item.purgeOnUse && p.hasStatus<PS::DUPLICATION>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::DUPLICATION>();
    }

    const auto echoForm = p.getStatus<PS::ECHO_FORM>();
    if (!item.purgeOnUse && echoForm) {
        const bool echoFormActive = player.cardsPlayedThisTurn - player.echoFormCardsDoubled <= echoForm;
        if (echoFormActive) {
            ++player.echoFormCardsDoubled;
            queuePurgeCard(c, item.target);
        }
    }

    if (p.hasStatus<PS::HEX>()) {
        Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::DAZED), 1, true);
    }

    if (p.hasStatus<PS::PANACHE>() && --p.panacheCounter <= 0) {
        Actions::DamageAllEnemy(*this, p.getStatus<PS::PANACHE>());
    }

    // todo Storm
    // todo Heatsinks
    // todo BirdFacedUrn

    // ********* Relics onUseCard *********
    // todo ink bottle/ ornamental fan need to be ordered i believe

    if (p.hasRelic<R::INK_BOTTLE>()) {
        p.inkBottleCounter++;
        if (p.inkBottleCounter == 10) {
            p.inkBottleCounter = 0;
            Actions::DrawCards(*this, 1);
        }
    }

    if (p.hasRelic<R::ORANGE_PELLETS>()) {
        p.orangePelletsCardTypesPlayed.set(static_cast<int>(CardType::SKILL), true); // set bit 0 true
        if (p.orangePelletsCardTypesPlayed.all()) {
            p.orangePelletsCardTypesPlayed.reset();
            Actions::RemovePlayerDebuffs(*this);
        }
    }

    if (p.hasRelic<R::LETTER_OPENER>()) {
        if (p.skillsPlayedThisTurn >= 3 &&  p.skillsPlayedThisTurn % 3 == 0) {
            Actions::DamageAllEnemy(*this, 5);
        }
    }

    if (p.hasRelic<R::MUMMIFIED_HAND>()) {
        // todo
    }

    /*
     *  for each card in hand : TriggerOnCardPlayed
     *  for each card in discardPile : TriggerOnCardPlayed
     *  for each card in drawPile : TriggerOnCardPlayed
     */

    // ********* Enemy Powers onUseCard *********
    // todo Choke Power

    auto &m = monsters.arr[0];
    if (m.hasStatus<MS::ENRAGE>()) {
        m.buff<MS::STRENGTH>(m.getStatus<MS::ENRAGE>());
    }
}

void BattleContext::onUsePowerCard() {
    auto &item = currentCard;
    auto &c = item.card;
    auto &p = player;

    cards.reduceForceFieldCosts();

    if (p.hasStatus<PS::AFTER_IMAGE>()) {
        Actions::GainBlock(*this, p.getStatus<PS::AFTER_IMAGE>());
    }

    if (!item.purgeOnUse && p.hasStatus<PS::AMPLIFY>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::AMPLIFY>();
    }

    if (!item.purgeOnUse && p.hasStatus<PS::DUPLICATION>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::DUPLICATION>();
    }

    const auto echoForm = p.getStatus<PS::ECHO_FORM>();
    if (!item.purgeOnUse && echoForm) {
        const bool echoFormActive = player.cardsPlayedThisTurn - player.echoFormCardsDoubled <= echoForm;
        if (echoFormActive) {
            ++player.echoFormCardsDoubled;
            queuePurgeCard(c, item.target);
        }
    }

    if (p.hasStatus<PS::HEX>()) {
        Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::DAZED), 1, true);
    }

    if (p.hasStatus<PS::PANACHE>() && --p.panacheCounter <= 0) {
        Actions::DamageAllEnemy(*this, p.getStatus<PS::PANACHE>());
    }

    for (int i = 0; i < p.storm; ++i) {
        Actions::ChannelOrb(*this, Orb::LIGHTNING);
    }
    if (p.heatsinks > 0) {
        Actions::DrawCards(*this, p.heatsinks);
    }

    // ********* Relics onUseCard *********

    if (p.hasRelic<R::BIRD_FACED_URN>()) {
        p.heal(2);
    }

    if (p.hasRelic<R::INK_BOTTLE>()) {
        p.inkBottleCounter++;
        if (p.inkBottleCounter == 10) {
            p.inkBottleCounter = 0;
            Actions::DrawCards(*this, 1);
        }
    }

    if (p.hasRelic<R::ORANGE_PELLETS>()) {
        p.orangePelletsCardTypesPlayed.set(static_cast<int>(CardType::POWER), true); // set bit 0 true
        if (p.orangePelletsCardTypesPlayed.all()) {
            p.orangePelletsCardTypesPlayed.reset();
            Actions::RemovePlayerDebuffs(*this);
        }
    }

    if (p.hasRelic<R::MUMMIFIED_HAND>()) {
        mummifiedHandOnUsePower();
    }

//    auto &m = monsters.optionMap[2];
//    if (m.hasStatusInternal<MS::CURIOSITY>()) {
//        m.buff<MS::STRENGTH>(m.getStatus<MS::CURIOSITY>());
//    }
}

void BattleContext::onUseStatusOrCurseCard() {
    auto &item = currentCard;
    auto &c = item.card;
    auto &p = player;

    if (p.hasStatus<PS::AFTER_IMAGE>()) {
        Actions::GainBlock(*this, p.getStatus<PS::AFTER_IMAGE>());
    }

    if (!item.purgeOnUse && p.hasStatus<PS::DUPLICATION>()) {
        queuePurgeCard(c, item.target);
        p.decrementStatus<PS::DUPLICATION>();
    }

    const auto echoForm = p.getStatus<PS::ECHO_FORM>();
    if (!item.purgeOnUse && echoForm) {
        const bool echoFormActive = player.cardsPlayedThisTurn - player.echoFormCardsDoubled <= echoForm;
        if (echoFormActive) {
            ++player.echoFormCardsDoubled;
            queuePurgeCard(c, item.target);
        }
    }

    if (p.hasStatus<PS::HEX>()) {
        Actions::MakeTempCardInDrawPile(*this, CardInstance(CardId::DAZED), 1, true);
    }

    if (p.hasStatus<PS::PANACHE>() && --p.panacheCounter <= 0) {
        Actions::DamageAllEnemy(*this, p.getStatus<PS::PANACHE>());
    }

    if (c.getType() == CardType::CURSE) {
        if (p.hasRelic<R::BLUE_CANDLE>()) {
            Actions::PlayerLoseHp(*this, 1, true);
            item.exhaustOnUse = true;
        }

    } else if (c.getType() == CardType::STATUS) {
        if (p.hasRelic<R::MEDICAL_KIT>()) {
            item.exhaustOnUse = true;
        }
    }

    if (p.hasRelic<R::INK_BOTTLE>()) {
        p.inkBottleCounter++;
        if (p.inkBottleCounter == 10) {
            p.inkBottleCounter = 0;
            Actions::DrawCards(*this, 1);
        }
    }

}

void BattleContext::onAfterUseCard() {
    auto &item = currentCard;
    auto &c = item.card;

    if (item.triggerOnUse) {
        auto &m = monsters.arr[0];
        if (m.hasStatus<MS::TIME_WARP>()) {
            auto timeWarp = m.getStatus<MS::TIME_WARP>();
            if (timeWarp == 11) {
                m.setStatus<MS::TIME_WARP>(0);
                m.buff<MS::STRENGTH>(2);
                callEndTurnEarlySequence();

            } else {
                m.setStatus<MS::TIME_WARP>(timeWarp + 1);
                ++timeWarp;
            }
        }
        if (m.hasStatus<MS::SLOW>()) {
            m.buff<MS::SLOW>(1);
        }
        if (m.hasStatus<MS::BEAT_OF_DEATH>()) {
            Actions::DamagePlayer(*this, m.getStatus<MS::BEAT_OF_DEATH>());
        }
    }

    if (item.purgeOnUse) {
        return;
    }

    bool rebound = false;
    c.freeToPlayOnce = false;

    if (c.getType() == CardType::POWER) {
        c.id = CardId::INVALID;
        if (endTurnAfterCurrentCard) {
            endTurnAfterCurrentCard = false;
            endTurn();
        }
        return;
    }

    if (player.hasStatus<PS::REBOUND>()) {
        if (player.getStatus<PS::REBOUND>() == 1) {
            rebound = true;
            player.setHasStatus<PS::REBOUND>(false);

        } else {
            player.setStatusValueNoChecks<PS::REBOUND>(1);
        }
    }

    bool spoonProc = false;
    if (item.exhaustOnUse && player.hasRelic<R::STRANGE_SPOON>()) {
        spoonProc = cardRandomRng.randomBoolean();
    }

    if (item.exhaustOnUse && !spoonProc) {
        triggerAndMoveToExhaustPile(c);

    } else {
        // targetCard.exhaustOnUseOnce = false;
        // targetCard.dontTriggerOnUseCard = false;

        if (rebound) { // these methods all call resetCardBeforeMoving but it looks unnecessary
            cards.moveToDrawPileTop(c);

        } else if (c.id == CardId::TANTRUM) {
            cards.shuffleIntoDrawPile(cardRandomRng, c);

        } else {
            // The game calls OnCardDrawOrDiscard here which only does two things:
            // 1. sets the damage on all shivs in hand if you have accuracy power,
            // 2. sets the cost of all skills in hand to -9 if you have corruption
            // we will handle these tasks elsewhere
            cards.moveToDiscardPile(c);
        }
    }
    if (endTurnAfterCurrentCard) {
        endTurnAfterCurrentCard = false;
        endTurn();
    }
    // TODO these must be done in the cards method itself
    // todo make Accuracy part of calculateBaseDamage

    // this.targetCard.exhaustOnUseOnce = false;
    //            this.targetCard.dontTriggerOnUseCard = false;
}

// **********************

void BattleContext::endTurn() {
    endTurnRequested = true;
    energyWasted += player.energy;
    for (auto& frame : resolution) {
        if (!frame.isCard()) frame.phase = ResolutionPhase::Done;
        else if (frame.phase != ResolutionPhase::CardDone) frame.phase = ResolutionPhase::CardCleanup;
    }
}

void BattleContext::callEndOfTurnActions() {

    // ********************* Player Relics OnPlayerEndTurn *********************

    if (player.hasRelic<R::CLOAK_CLASP>()) {
        Actions::GainBlock(*this, cards.cardsInHand);
    }

    if (player.hasRelic<R::FROZEN_CORE>()) {
        if (player.hasEmptyOrb()) {
            player.channelOrb(*this, Orb::FROST);
        }
    }

    if (player.hasRelic<R::NILRYS_CODEX>()) {
        Actions::CodexAction(*this);
    }

    if (player.hasRelic<R::ORICHALCUM>()) {
        if (player.block <= 0) {
            Actions::GainBlock(*this, 6);
        }
    }

    if (player.hasRelic<R::STONE_CALENDAR>()) {
        if (turn == 6) {
            Actions::DamageAllEnemy(*this, 52);
        }
    }

    // ********************* Player Powers AtEndOfTurnPreEndTurnCards *********************

    if (player.hasStatus<PS::METALLICIZE>()) {
        Actions::GainBlock(*this,  player.getStatus<PS::METALLICIZE>());
    }

    if (player.hasStatus<PS::PLATED_ARMOR>()) {
        Actions::GainBlock(*this, player.getStatus<PS::PLATED_ARMOR>());
    }

    if (player.hasStatus<PS::LIKE_WATER>() && player.stance == Stance::CALM) {
        Actions::GainBlock(*this, player.getStatus<PS::LIKE_WATER>());
    }

    if (player.orbSlots) {
        Actions::TriggerEndOfTurnOrbsAction(*this);
    }

    // todo for cards in hand call triggerOnEndOfTurnForPlayingCard

    for (int i = 0; i < cards.cardsInHand; ++i) {

        const auto &c = cards.hand[i];

        switch (c.id) {
            case CardId::BURN:
            case CardId::DECAY:
            case CardId::DOUBT:
            case CardId::SHAME:
            case CardId::REGRET: {
                CardPlay item;
                item.triggerOnUse = false;
                item.regretCardCount = cards.cardsInHand;
                item.card = c;
                beginCard(item);
                break;
            }
            default:
                break;
        }
    }

    // todo stance onEndOfTurn
}

void BattleContext::onTurnEnding() {
    // AbstractRoom endTurn()

    // ********************* Player Powers atEndOfTurn *********************
    player.applyEndOfTurnPowers(*this);

    // todo reset card attributes here in draw, discard and hand
    cards.resetAttributesAtEndOfTurn();

}

void BattleContext::callEndTurnEarlySequence() {
    endTurnAfterCurrentCard = true;
}

void BattleContext::applyEndOfRoundPowers() {
    for (int i = 0; i < monsters.monsterCount; i++) {
        auto &m = monsters.arr[i];
        if (m.isDying() || m.isEscaping()) {
            continue;
        }
        m.applyEndOfTurnTriggers(*this);
    }

    player.applyAtEndOfRoundPowers();

    for (int i = 0; i < monsters.monsterCount; i++) {
        auto &m = monsters.arr[i];
        if (m.isDying() || m.isEscaping()) {
            continue;
        }
        m.applyEndOfRoundPowers(*this);
    }
}

void BattleContext::afterMonsterTurns() {
    // ********* Enemy powers atEndOfRound *********
    if (!skipMonsterTurn) {
        applyEndOfRoundPowers();
    }

    ++turn;
    skipMonsterTurn = false;
    turnHasEnded = false;

    // player stance atStartOfTurn
    if (player.stance == Stance::DIVINITY) {
        Actions::ChangeStance(*this, Stance::NEUTRAL);
    }

    player.applyStartOfTurnRelics(*this);

    // player applyStartOfTurnPreDrawCards() // no cards implement this
    // player.applyStartOfTurnCards() // only Eviscerate implements this

}

void BattleContext::finishStartTurn() {
    player.triggerStartOfTurnOrbs(*this);

    if (player.hasStatus<PS::BARRICADE>()) {

    } else if (player.hasStatus<PS::BLUR>()) {
        player.decrementStatus<PS::BLUR>();

    } else if (player.hasRelic<R::CALIPERS>()) {
        player.block = std::max(0, player.block-15);

    } else {
        player.block = 0;
    }

    if (isBattleOver) {
        return;
    }

    Actions::DrawCards(*this, player.cardDrawPerTurn); // in this action, an effect queue item is added to rechard energy lol

    if (player.hasStatus<PS::DRAW_REDUCTION>()) {
        if (player.wasJustApplied<PS::DRAW_REDUCTION>()) {
            player.setJustApplied<PS::DRAW_REDUCTION>(false);
        } else {
            player.removeStatus<PS::DRAW_REDUCTION>();
            ++player.cardDrawPerTurn;
        }
    }

    player.applyStartOfTurnPostDrawRelics(*this);
    player.applyStartOfTurnPostDrawPowers(*this);

    // this has to be here because some relics check this info.
    player.cardsPlayedThisTurn = 0;
    player.attacksPlayedThisTurn = 0;
    player.skillsPlayedThisTurn = 0;
    player.cardsDiscardedThisTurn = 0;

    player.rechargeEnergy(*this); // this is called by the PlayerTurnEffect in game, I think it can be done here
}

void BattleContext::obtainPotion(Potion p) {
    if (potionCount == potionCapacity || player.hasRelic<R::SOZU>()) {
        return; // no empty potion slots
    }

    for (int i = 0; i < potionCapacity; ++i) {
        if (potions[i] == Potion::EMPTY_POTION_SLOT) {
            ++potionCount;
            potions[i] = p;
            return;
        }
    }

//  todo prevent this, do nothing for now
//    assert(false); // unreachable
}

void BattleContext::discardPotion(int idx) {
    potions[idx] = Potion::EMPTY_POTION_SLOT;
    --potionCount;
}

void BattleContext::drinkPotion(int idx, int target) {
    const bool hasBark = player.hasRelic<R::SACRED_BARK>();
    const Potion p = potions[idx];

    // Search actions enforce this too, but keep direct simulator callers from
    // consuming a potion or escaping through an impossible boss transition.
    // Smoke Bomb is disabled: escaping was scored as PLAYER_VICTORY, which
    // search treats as a win. It is never generated (see getRandomPotion).
    if (p == Potion::SMOKE_BOMB) {
        return;
    }

    discardPotion(idx);

    if (player.hasRelic<R::TOY_ORNITHOPTER>()) {  // heal 5 whenever a potion is used
        player.heal(5);
    }


    switch (p) {
        case Potion::AMBROSIA:
            Actions::ChangeStance(*this, Stance::DIVINITY);
            break;

        case Potion::ANCIENT_POTION:
            Actions::BuffPlayer<PS::ARTIFACT>(*this, hasBark ? 2 : 1);
            break;

        case Potion::ATTACK_POTION:
            Actions::DiscoveryAction(*this, CardType::ATTACK, hasBark ? 2 : 1);
            break;

        case Potion::BLESSING_OF_THE_FORGE:
            Actions::UpgradeAllCardsInHand(*this);
            break;

        case Potion::BLOCK_POTION:
            Actions::GainBlock(*this, hasBark ? 24 : 12);
            break;


        case Potion::BLOOD_POTION: {
            int healAmt = static_cast<int>((static_cast<float>(player.maxHp * (hasBark ? 40 : 20)) / 100.0f));
            Actions::HealPlayer(*this, healAmt);
            break;
        }

        case Potion::BOTTLED_MIRACLE:
            Actions::MakeTempCardInHand(*this, CardId::MIRACLE, false, hasBark ? 4 : 2);
            break;

        case Potion::COLORLESS_POTION:
            Actions::DiscoveryAction(*this, CardType::STATUS, hasBark ? 2 : 1); // status card type is being used to indicate colorless
            break;

        case Potion::CULTIST_POTION:
            Actions::BuffPlayer<PS::RITUAL>(*this, hasBark ? 2 : 1);
            break;

        case Potion::CUNNING_POTION:
            Actions::MakeTempCardInHand(*this, CardId::SHIV, true, hasBark ? 6 : 3);
            break;

        case Potion::DEXTERITY_POTION:
            Actions::BuffPlayer<PS::DEXTERITY>(*this, hasBark ? 4 : 2);
            break;

        case Potion::DISTILLED_CHAOS: {
            const auto cardsToPlay = hasBark ? 6 : 3;
            for (int i = 0; i < cardsToPlay; ++i) {
                Actions::PlayTopCard(*this, monsters.getRandomMonsterIdx(cardRandomRng), false);
            }
            break;
        }

        case Potion::DUPLICATION_POTION:
            Actions::BuffPlayer<PS::DUPLICATION>(*this, hasBark ? 2 : 1);
            break;

        case Potion::ELIXIR_POTION:
            Actions::ExhaustMany(*this, 10);
            break;

        case Potion::ENERGY_POTION:
            Actions::GainEnergy(*this, hasBark ? 4 : 2);
            break;

        case Potion::ENTROPIC_BREW: {
            for (int i = 0; i < potionCapacity; ++i) {
                Potion randomPotion = returnRandomPotion(potionRng, player.cc, true);
                obtainPotion(randomPotion);
            }
            break;
        }

        case Potion::ESSENCE_OF_DARKNESS:
            Actions::EssenceOfDarkness(*this, hasBark ? 2 : 1);
            break;

        case Potion::ESSENCE_OF_STEEL:
            Actions::BuffPlayer<PS::PLATED_ARMOR>(*this, hasBark ? 8 : 4);
            break;

        case Potion::EXPLOSIVE_POTION: {
            const auto damage = hasBark ? 20 : 10;
            Actions::DamageAllEnemy(*this, damage); // todo does not having player be owner here matter?
            break;
        }

        case Potion::FEAR_POTION:
            Actions::DebuffEnemy<MS::VULNERABLE>(*this, target, hasBark ? 6 : 3, false);
            break;

        case Potion::FIRE_POTION:
            Actions::DamageEnemy(*this, target, hasBark ? 40 : 20);
            break;

        case Potion::FLEX_POTION:
            Actions::BuffPlayer<PS::STRENGTH>(*this, hasBark ? 10 : 5);
            Actions::DebuffPlayer<PS::LOSE_STRENGTH>(*this, hasBark ? 10 : 5);
            break;

        case Potion::FOCUS_POTION:
            Actions::BuffPlayer<PS::FOCUS>(*this, hasBark ? 4 : 2);
            break;

        case Potion::FRUIT_JUICE:
            player.increaseMaxHp(hasBark ? 10 : 5);
            break;

        case Potion::GAMBLERS_BREW:
            Actions::GambleAction(*this);
            break;

        case Potion::GHOST_IN_A_JAR:
            Actions::BuffPlayer<PS::INTANGIBLE>(*this, hasBark ? 2 : 1);
            break;

        case Potion::HEART_OF_IRON:
            Actions::BuffPlayer<PS::METALLICIZE>(*this, hasBark ? 12 : 6);
            break;

        case Potion::LIQUID_BRONZE:
            Actions::BuffPlayer<PS::THORNS>(*this, hasBark ? 6 : 3);
            break;

        case Potion::LIQUID_MEMORIES:
            Actions::BetterDiscardPileToHandAction(*this, hasBark ? 2 : 1, CardSelectTask::LIQUID_MEMORIES_POTION);
            break;

        case Potion::POISON_POTION:
            Actions::DebuffEnemy<MS::POISON>(*this, target, hasBark ? 12 : 6);
            break;

        case Potion::POTION_OF_CAPACITY:
            Actions::IncreaseOrbSlots(*this, hasBark ? 4 : 2);
            break;

        case Potion::POWER_POTION:
            haveUsedDiscoveryAction = true;
            Actions::DiscoveryAction(*this, CardType::POWER, hasBark ? 2 : 1);
            break;

        case Potion::REGEN_POTION:
            Actions::BuffPlayer<PS::REGEN>(*this, hasBark ? 10 : 5);
            break;

        case Potion::SKILL_POTION:
            Actions::DiscoveryAction(*this, CardType::SKILL, hasBark ? 2 : 1);
            break;

        case Potion::SMOKE_BOMB:
            escapedCombat = true;
            outcome = Outcome::PLAYER_VICTORY;
            break;

        case Potion::SNECKO_OIL:
            Actions::DrawCards(*this, hasBark ? 10 : 5);
            Actions::RandomizeHandCost(*this);
            break;

        case Potion::SPEED_POTION:
            Actions::BuffPlayer<PS::DEXTERITY>(*this, hasBark ? 10 : 5);
            Actions::DebuffPlayer<PS::LOSE_DEXTERITY>(*this, hasBark ? 10 : 5);
            break;

        case Potion::STANCE_POTION:
            Actions::SetState(*this, InputState::CHOOSE_STANCE_ACTION);
            break;

        case Potion::STRENGTH_POTION:
            Actions::BuffPlayer<PS::STRENGTH>(*this, hasBark ? 4 : 2);
            break;

        case Potion::SWIFT_POTION:
            Actions::DrawCards(*this, hasBark ? 6 : 3);
            break;

        case Potion::WEAK_POTION:
            Actions::DebuffEnemy<MS::WEAK>(*this, target, hasBark ? 6 : 3, false);
            break;

        case Potion::INVALID:
        case Potion::EMPTY_POTION_SLOT:
        case Potion::FAIRY_POTION:
        default:
            std::cerr << seed << "invalid drink potion: " << static_cast<int>(p) << std::endl;
            assert(false);
            break;
    }

}

void BattleContext::noteDrawPileOrderAccess() {
    // Known top cards and previously scried cards may be conservatively counted as hidden.
    if (!player.hasRelic<R::FROZEN_EYE>()) ++hiddenInformationAccesses;
}

void BattleContext::noteEnemyIntentAccess() {
    if (player.hasRelic<R::RUNIC_DOME>()) ++hiddenInformationAccesses;
}

void BattleContext::drawCards(int count) {
    if (count <= 0 || player.hasStatus<PS::NO_DRAW>()) return;
    while (count-- > 0 && cards.cardsInHand < CardManager::MAX_HAND_SIZE && outcome == Outcome::UNDECIDED) {
        if (cards.drawPile.empty()) {
            if (cards.discardPile.empty()) break;
            onShuffle();
            Actions::EmptyDeckShuffle(*this);
        }
        if (cards.drawPile.empty()) break;
        ++cardsDrawn;
        cards.draw(*this, 1);
    }
}

void BattleContext::discardAtEndOfTurn() {
    std::vector<CardInstance> original(cards.hand.begin(), cards.hand.begin() + cards.cardsInHand);
    for (auto c : original) {
        if (!c.isEthereal()) continue;
        cards.removeFromHandById(c.uniqueId);
        triggerAndMoveToExhaustPile(c);
    }
    if (player.hasRelic<R::RUNIC_PYRAMID>() || player.hasStatus<PS::EQUILIBRIUM>()) return;
    original.assign(cards.hand.begin(), cards.hand.begin() + cards.cardsInHand);
    for (auto c : original) {
        if (c.hasSelfRetain() || c.retain) continue;
        cards.removeFromHandById(c.uniqueId);
        cards.moveToDiscardPile(c);
        ++player.cardsDiscardedThisTurn;
    }
    for (int i = 0; i < cards.cardsInHand; ++i) cards.hand[i].retain = false;
}


void BattleContext::playTopCardInDrawPile(int monsterTargetIdx, bool exhausts) {
    if (cards.drawPile.empty() && !cards.discardPile.empty()) Actions::EmptyDeckShuffle(*this);
    if (cards.drawPile.empty()) return;
    noteDrawPileOrderAccess();
    CardPlay item{cards.popFromDrawPile(), monsterTargetIdx, player.energy};
    item.exhaustOnUse = exhausts;
    item.autoplay = true;
    item.freeToPlay = true;
    beginCard(item);
}

void BattleContext::moveToHandHelper(CardInstance c) {
    if (cards.cardsInHand < 10) {
        if (player.hasStatus<PS::CORRUPTION>() && c.getType() == CardType::SKILL) {
            c.setCostForTurn(-9);
        }
        cards.moveToHand(c);
    } else {
        cards.moveToDiscardPile(c);
    }
}

void BattleContext::exhaustSpecificCardInHand(int idx, std::int16_t uniqueId) {

    int foundIdx = -1;

    if (idx < cards.cardsInHand && cards.hand[idx].uniqueId == uniqueId) {
        foundIdx = idx;
    } else {
        for (int i = 0; i < cards.cardsInHand; ++i) {
            if (cards.hand[i].uniqueId == uniqueId) {
               foundIdx = i;
               break;
            }
        }
    }

    if (foundIdx == -1) {
        std::cerr << "exhaustSpecificCardInHand: card not found in hand\n";
        return;
    }

    const auto card = cards.hand[foundIdx];
    cards.removeFromHandAtIdx(foundIdx);
    triggerAndMoveToExhaustPile(card);
}


void BattleContext::exhaustTopCardInHand() {
    if (cards.cardsInHand <= 0) {
        std::cerr << "exhaustTopCardInHand: no cards in hand";
        return;
    }

    auto &c = cards.hand[--cards.cardsInHand];
    cards.notifyRemoveFromHand(c);
    triggerAndMoveToExhaustPile(c);
}

//void BattleContext::incrementDiscard() { // not for end of turn
//    player.cardsDiscardedThisTurn++;
//    if (!turnHasEnded) {
//        // todo
//    }
//
////    ++totalDiscardedThisTurn;
////    if (!AbstractDungeon.actionManager.turnHasEnded && !endOfTurn) {
////        AbstractDungeon.player.updateCardsOnDiscard();
////        Iterator var1 = AbstractDungeon.player.relics.iterator();
////
////        while(var1.hasNext()) {
////            AbstractRelic r = (AbstractRelic)var1.next();
////            r.onManualDiscard();
////        }
////    }
//}

void BattleContext::triggerOnEndOfTurnForPlayingCards() {
//
//    bool foundCurse;
//    do {
//        foundCurse = false;
//
//        for (int i = 0; i <= hand.size(); ++i) {
//            switch (hand[i].id) {
//                case CardId::DECAY:
//                    Actions::DamagePlayer(*this, 2);
//                    foundCurse = true;
//                    break;
//
//                case CardId::DOUBT:
//                    Actions::DebuffPlayer<PS::WEAK>(*this, 1);
//                    foundCurse = true;
//                    break;
//
//                case CardId::SHAME:
//                    Actions::DebuffPlayer<PS::FRAIL>(*this, 1);
//                    foundCurse = true;
//                    break;
//
//                case CardId::REGRET:
//                    Actions::PlayerLoseHp(*this, hand.size(), true);
//                    foundCurse = true;
//                    break;
//
//                case CardId::BURN:
//                    Actions::DamagePlayer(*this, hand[i].upgraded ? 4 : 2);
//                    foundCurse = true;
//                    break;
//
//                default:
//                    break;
//            }
//
//            if (foundCurse) {
//                hand.removeCardAtIdx(i);
//            }
//
//        }
//
//    } while (foundCurse);

}

void BattleContext::triggerOnOtherCardPlayed(const CardInstance &usedCard) {
    int painCount = cards.handPainCount;
    if (usedCard.getId() == CardId::PAIN) {
        --painCount;
    }
    for (int i = 0; i < painCount; ++i) {
        Actions::PlayerLoseHp(*this, 1);
    }

    const auto thousandCuts = player.getStatus<PS::THOUSAND_CUTS>();
    if (thousandCuts) {
        Actions::DamageAllEnemy(*this, thousandCuts);
    }
}

int BattleContext::calculateCardDamage(const CardInstance &card, int targetIdx, int baseDamage) const {

    auto damage = static_cast<float>(baseDamage);

    // ****** Player Relics AtDamageModify ******

    if (player.hasRelic<R::STRIKE_DUMMY>() && card.isStrikeCard()) {
        damage += 3;
    }

    if (player.hasRelic<R::WRIST_BLADE>() && card.costForTurn == 0) {
        damage += 4;
    }


    // ****** Player Powers AtDamageGive ******

    damage += static_cast<float>(player.getStatus<PS::STRENGTH>());

    if (card.id == CardId::SHIV) {
        damage += static_cast<float>(player.getStatus<PS::ACCURACY>());
    }
    if (player.hasStatus<PS::VIGOR>()) {
        damage += static_cast<float>(player.getStatus<PS::VIGOR>());
    }

    if (player.hasStatus<PS::DOUBLE_DAMAGE>()) {
        damage *= 2;
    }

    if (player.hasStatus<PS::PEN_NIB>()) {
        damage *= 2;
    }

    if (player.hasStatus<PS::WEAK>()) {
        damage *= .75f;
    }

    // ****** Stance AtDamageGive ******

    if (player.stance == Stance::WRATH) {
        damage *= 2;
    } else if (player.stance == Stance::DIVINITY) {
        damage *= 3;
    }

    // ****** Enemy Powers AtDamageReceive ******
    const Monster &monster = monsters.arr[targetIdx];

    if (monster.hasStatus<MS::SLOW>()) {
        damage *= 1 + static_cast<float>(monster.getStatus<MS::SLOW>()) * 0.1f;
    }

    if (monster.hasStatus<MS::VULNERABLE>()) {
        if (player.hasRelic<R::PAPER_PHROG>()) {
            damage *= 1.75f;
        } else {
            damage *= 1.5f;
        }
    }


    // ****** Player Powers AtDamageGiveFinal ****** (none ?)
    // ****** Monster Powers AtDamageReceiveFinal ******

    if (monster.hasStatus<MS::FLIGHT>()) {
        damage *= .5;
    }

    if (monster.hasStatus<MS::INTANGIBLE>()) {
        damage = std::max(damage, 1.0f);
    }

    return std::max(0, static_cast<int>(damage));
}

int BattleContext::calculateCardBlock(int baseBlock) const {
    if (player.hasStatus<PS::NO_BLOCK>()) {
        return 0;
    }

    int block = baseBlock;
    if (player.hasStatus<PS::DEXTERITY>()) {
        block = std::max(0, block + player.getStatus<PS::DEXTERITY>());
    }

    if (player.hasStatus<PS::FRAIL>()) {
        return block * 3 / 4;
    }

    return block;
}

void BattleContext::queuePurgeCard(const CardInstance &c, int target) {
    CardPlay item;
    item.card = c;
    item.purgeOnUse = true;
    item.target = target;
    item.energyOnUse = currentCard.energyOnUse;
    item.ignoreEnergyTotal = true;
    item.autoplay = true;
    beginCard(item);
}

void BattleContext::noOpRollMove() {
    aiRng.random(99);
}

void BattleContext::onManualDiscard(const CardInstance &c) {
    if (c.getId() == CardId::TACTICIAN) {
        player.gainEnergy(c.isUpgraded() ? 2 : 1);
    } else if (c.getId() == CardId::REFLEX) {
        Actions::DrawCards(*this, c.isUpgraded() ? 3 : 2);
    }
}

void BattleContext::onShuffle() {
    if (player.hasRelic<R::THE_ABACUS>()) {
        Actions::GainBlock(*this, 6);
    }

    if (player.hasRelic<R::MELANGE>()) {
//        Actions::SetState(*this, InputState::SCRY, 3); // TODO SCRY Action
    }

    if (player.hasRelic<R::SUNDIAL>()) {
        if (player.sundialCounter == 2) {
            player.sundialCounter = 0;
            Actions::GainEnergy(*this, 2);
        } else {
            ++player.sundialCounter;
        }
    }
}

void BattleContext::triggerAndMoveToExhaustPile(CardInstance c) {
    // player relics onExhaust
    // player powers onExhaust
    // (the card).triggerOnExhaust

    if (player.hasRelic<R::CHARONS_ASHES>()) {
        Actions::DamageAllEnemy(*this, 3);
    }

    if (player.hasRelic<R::DEAD_BRANCH>()){
        CardId id = getTrulyRandomCardInCombat(cardRandomRng, player.cc);
        Actions::MakeTempCardInHand(*this, id);
    }

    if (player.hasStatus<PS::DARK_EMBRACE>()) {
        Actions::DrawCards(*this, player.getStatus<PS::DARK_EMBRACE>());
    }

    if (player.hasStatus<PS::FEEL_NO_PAIN>()) {
        Actions::GainBlock(*this, player.getStatus<PS::FEEL_NO_PAIN>());
    }

    if (c.getId() == CardId::NECRONOMICURSE) {
        Actions::MakeTempCardInHand(*this, CardId::NECRONOMICURSE);
    }

    if (c.getId() == CardId::SENTINEL) {
        player.gainEnergy(c.isUpgraded() ? 3 : 2); // the game adds to bot here
    }

    cards.moveToExhaustPile(c);
}

void BattleContext::mummifiedHandOnUsePower() {
    fixed_list<int,10> matchingIdxList;

    for (int i = 0; i < cards.cardsInHand; ++i) {
        const auto &c = cards.hand[i];
        bool canPick = c.cost > 0 && c.costForTurn > 0 && !c.freeToPlayOnce;
        if (canPick) {
            matchingIdxList.push_back(i);
        }
    }

    if (matchingIdxList.empty()) {
        return;
    }

    for (int i = matchingIdxList.size()-1; i >= 0; --i) {
        const auto uniqueId = cards.hand[matchingIdxList[i]].getUniqueId();
        if (std::any_of(resolution.begin(), resolution.end(), [uniqueId](const auto& frame) {
            return frame.isCard() && frame.card.card.uniqueId == uniqueId;
        })) {
            matchingIdxList.remove(i);
        }
    }

    if (matchingIdxList.empty()) {
        return;
    }

    const int selectedListIdx = cardRandomRng.random(0,matchingIdxList.size()-1);
    const int selectedHandIdx = matchingIdxList[selectedListIdx];
    cards.hand[selectedHandIdx].setCostForTurn(0);
}

void BattleContext::openDiscoveryScreen(std::array<CardId, 3> discoveryCards, int copyCount) {
    inputState = InputState::CARD_SELECT;
    cardSelectInfo.cardSelectTask = CardSelectTask::DISCOVERY;
    cardSelectInfo.pickCount = 1;
    cardSelectInfo.canPickAnyNumber = false;
    cardSelectInfo.canPickZero = false;
    cardSelectInfo.cards = discoveryCards;
    cardSelectInfo.discovery_CopyCount() = copyCount;
}

void BattleContext::openSimpleCardSelectScreen(CardSelectTask task, int count) {
    inputState = InputState::CARD_SELECT;
    cardSelectInfo.cardSelectTask = task;
    cardSelectInfo.pickCount = count;
    cardSelectInfo.canPickAnyNumber = false;
    cardSelectInfo.canPickZero = false;
}


void BattleContext::chooseArmamentsCard(int handIdx) {
    // todo cleaner solution

    fixed_list<CardInstance,10> validCards;
    fixed_list<CardInstance,10> invalidCards;
    for (int i = 0; i < cards.cardsInHand; ++i) {
        const auto &c = cards.hand[i];
        if (i == handIdx) {
            continue;
        }
        if (c.canUpgrade()) {
            validCards.push_back(c);
        } else {
            invalidCards.push_back(c);
        }
    }

    CardInstance cardToUpgrade = cards.hand[handIdx];
    cardToUpgrade.upgrade();

    int i = 0;
    for (auto c : validCards) {
        cards.hand[i++] = c;
    }
    cards.hand[i++] = cardToUpgrade;
    for (auto c : invalidCards) {
        cards.hand[i++] = c;
    }

}

void BattleContext::chooseCodexCard(CardId id) {
    CardInstance c(id);
    c.uniqueId = static_cast<std::int16_t>(cards.nextUniqueCardId++);
    cards.notifyAddCardToCombat(c);
    cards.shuffleIntoDrawPile(cardRandomRng, c);
}

void BattleContext::chooseDualWieldCard(int handIdx) {

    // dual wield is so fucking buggy
    // if you dual wield a ritual dagger:
    // when there is no choice on which card to pick, the first one will change the card in the deck
    // when there **is** a choice on which card to pick, neither will change the card in the deck XDD

    const int copyCount = cardSelectInfo.dualWield_CopyCount();
    CardInstance dualWieldCard = cards.hand[handIdx];

    // todo cleaner solution

    fixed_list<CardInstance,10> validCards;
    fixed_list<CardInstance,10> invalidCards;
    for (int i = 0; i < cards.cardsInHand; ++i) {
        const auto &c = cards.hand[i];
        if (i == handIdx) {
            continue;
        }
        if (c.getType() == CardType::ATTACK || c.getType() == CardType::POWER) {
            validCards.push_back(c);
        } else {
            invalidCards.push_back(c);
        }
    }


    int i = 0;
    for (auto c : validCards) {
        cards.hand[i++] = c;
    }
    for (auto c : invalidCards) {
        cards.hand[i++] = c;
    }

    dualWieldCard.uniqueId = static_cast<std::int16_t>(cards.nextUniqueCardId++); // dual wield buggy
    cards.hand[i++] = dualWieldCard;

    for (int x = 0; x < copyCount; ++x) {
        if (cards.cardsInHand + 1 <= CardManager::MAX_HAND_SIZE) {
            cards.createTempCardInHand(dualWieldCard);

        } else {
            cards.createTempCardInDiscard(dualWieldCard);

        }
    }

}

void BattleContext::chooseDiscardToHandCard(int discardIdx, bool forZeroCost) {
    CardInstance c = cards.discardPile[discardIdx];
    cards.removeFromDiscard(discardIdx);
    // The one-card fast path never opens a selection screen; its task can be stale.
    if (forZeroCost) {
        c.setCostForTurn(0);
    }
    moveToHandHelper(c);
}

void BattleContext::chooseDiscoveryCard(CardId id) {
    const auto discoveryAmount = cardSelectInfo.data0;
    CardInstance c(id);
    c.setCostForTurn(0);

    for (int i = 0; i < discoveryAmount; ++i) {
        if (cards.cardsInHand + 1 <= CardManager::MAX_HAND_SIZE) {
            if (player.hasStatus<PS::CORRUPTION>() && c.getType() == CardType::SKILL) {
                c.setCostForTurn(-9);
            }
            cards.createTempCardInHand(c);

        } else {
            cards.createTempCardInDiscard(c);
        }
    }
}

void BattleContext::chooseExhaustOneCard(int handIdx) {
    auto c = cards.hand[handIdx];
    cards.removeFromHandAtIdx(handIdx);
    triggerAndMoveToExhaustPile(c);
}


void BattleContext::chooseExhumeCard(int exhaustIdx) {
    // todo game handles corruption here
    auto c = cards.exhaustPile[exhaustIdx];
    cards.removeFromExhaustPile(exhaustIdx);
    cards.notifyAddCardToCombat(c);

    moveToHandHelper(c);
}

void BattleContext::chooseForethoughtCard(int handIdx) {
    if (cards.hand[handIdx].cost > 0) {
        cards.hand[handIdx].freeToPlayOnce = true;
    }

    cards.insertToDrawPile(0, cards.hand[handIdx]);
    cards.removeFromHandAtIdx(handIdx);
}

void BattleContext::chooseDiscardCard(int handIdx) {
    auto c = cards.hand[handIdx];
    cards.removeFromHandAtIdx(handIdx);
    cards.moveToDiscardPile(c);
    onManualDiscard(c);
}

void BattleContext::chooseHeadbuttCard(int discardIdx) {
#ifdef sts_asserts
    assert(discardIdx >= 0 && discardIdx < cards.discardPile.size());
#endif
    cards.moveToDrawPileTop(cards.discardPile[discardIdx]);
    cards.removeFromDiscard(discardIdx);
}

void BattleContext::chooseRecycleCard(int handIdx) {
    assert(handIdx >= 0 && handIdx < cards.cardsInHand);
    const auto card = cards.hand[handIdx];
    player.gainEnergy(std::max(0, static_cast<int>(card.costForTurn)));
    cards.removeFromHandAtIdx(handIdx);
    triggerAndMoveToExhaustPile(card);
}

void BattleContext::chooseSeekCard(int drawIdx) {
    assert(drawIdx >= 0 && drawIdx < cards.drawPile.size());
    const auto card = cards.drawPile[drawIdx];
    cards.removeFromDrawPileAtIdx(drawIdx);
    moveToHandHelper(card);
    if (--cardSelectInfo.data0 > 0 && !cards.drawPile.empty()) {
        Actions::SetState(*this, InputState::CARD_SELECT);
    }
}

void BattleContext::chooseOmniscienceCard(int drawIdx) {
    assert(drawIdx >= 0 && drawIdx < cards.drawPile.size());
    const auto card = cards.drawPile[drawIdx];
    cards.removeFromDrawPileAtIdx(drawIdx);
    const int target = card.requiresTarget() ? player.lastTargetedMonster : 0;
    queuePurgeCard(card, target);
    queuePurgeCard(card, target);
}

void BattleContext::chooseNightmareCard(int handIdx) {
    assert(handIdx >= 0 && handIdx < cards.cardsInHand);
    nightmareCards.push_back(cards.hand[handIdx]);
    nightmareCopyCounts.push_back(cardSelectInfo.data0);
}

void BattleContext::chooseScryCards(const fixed_list<int, 10> &idxs) {
    auto selected = idxs;
    std::sort(selected.begin(), selected.end(), [](int a, int b) { return a > b; });
    for (const int idx : selected) {
        const auto card = cards.drawPile[idx];
        cards.removeFromDrawPileAtIdx(idx);
        cards.moveToDiscardPile(card);
    }
}

void BattleContext::chooseWish(int choice) {
    const bool upgraded = currentCard.card.isUpgraded();
    switch (choice) {
        case 0:
            Actions::BuffPlayer<PS::PLATED_ARMOR>(*this, upgraded ? 4 : 3);
            break;
        case 1:
            player.gainGold(*this, 25);
            break;
        case 2:
            Actions::BuffPlayer<PS::STRENGTH>(*this, upgraded ? 10 : 8);
            break;
        default:
            assert(false);
            break;
    }
}

void BattleContext::chooseSetupCard(int handIdx) {
    auto card = cards.hand[handIdx];
    card.setCostForTurn(0);
    cards.removeFromHandAtIdx(handIdx);
    cards.moveToDrawPileTop(card);
}

void BattleContext::chooseWarcryCard(int handIdx) {
#ifdef sts_asserts
    assert(handIdx >= 0 && handIdx < cards.cardsInHand);
#endif
    cards.moveToDrawPileTop(cards.hand[handIdx]);
    cards.removeFromHandAtIdx(handIdx);
}

void BattleContext::chooseDrawToHandCards(const int *idxs, int cardCount) {
    for (int i = 0; i < cardCount; ++i) {
        const auto drawIdx = idxs[i];
        auto c = cards.drawPile[drawIdx];
        cards.removeFromDrawPileAtIdx(drawIdx);
        moveToHandHelper(c);
    }
}

void BattleContext::chooseExhaustCards(const fixed_list<int, 10> &idxs) {
    if (idxs.empty()) {
        return;
    }
    auto listCopy = idxs;
    std::sort(listCopy.begin(), listCopy.end(), [](auto a, auto b) { return b < a; });

    // assume idxs is sorted in descending order
    for (const auto handIdx : listCopy) {
        auto c = cards.hand[handIdx];
        cards.removeFromHandAtIdx(handIdx);
        triggerAndMoveToExhaustPile(c);
    }
}

void BattleContext::chooseGambleCards(const fixed_list<int, 10> &idxs) {
    if (idxs.empty()) {
        return;
    }
    auto listCopy = idxs;
    std::sort(listCopy.begin(), listCopy.end(), [](auto a, auto b) { return b < a; });

    // assume idxs is sorted in descending order
    for (const auto handIdx : listCopy) {
        auto c = cards.hand[handIdx];
        cards.removeFromHandAtIdx(handIdx);
        cards.moveToDiscardPile(c);
        onManualDiscard(c);
    }
    Actions::DrawCards(*this, listCopy.size());
}


    void printRngCounters(std::ostream &os, const BattleContext &bc) {
        const std::string separator = " ";
        os << '\t';

        os << "aiRng: " << bc.aiRng.counter << separator;
        os << "cardRandomRng: " << bc.cardRandomRng.counter << separator;
        os << "shuffleRng: " << bc.shuffleRng.counter << separator;
        os << "miscRng: " << bc.miscRng.counter << separator;
        os << "monsterHpRng: " << bc.monsterHpRng.counter << separator;
        os << "potionRng: " << bc.potionRng.counter << separator;

        os << '\n';
    }

    void printPotions(std::ostream &os, const BattleContext &bc) {
        const auto s = "\n\t";
        os << "\t" << "potionCount: " << bc.potionCount;
        os << s << "potionCapacity: " << bc.potionCapacity;

        os << s << "{ ";
        for (int i = 0; i < bc.potionCapacity; ++i) {
            os << getPotionName(bc.potions[i]) << ", ";
        }
        os << "}\n";
    }

    std::ostream& operator<<(std::ostream &os, const BattleContext &bc) {
        os << "BattleContext: {\n";
        printPotions(os, bc);
        printRngCounters(os, bc);

        os << "	resolutionDepth: " << bc.resolution.size()
            << ", turn: " << bc.turn
            << ", ascension " << bc.ascension
            << ", loopCount: " << bc.loopCount
            << ", sum: " << bc.sum
            << ", seed: " << bc.seed
            << "\n";

        os << bc.monsters;
        os << bc.player;
        os << bc.cards;
        os << "}\n";
        return os;
    }


}
