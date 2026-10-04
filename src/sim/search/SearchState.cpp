#include "sim/search/SearchState.h"
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace sts::search {
namespace {
using Words = std::vector<std::uint64_t>;
template<class T> void add(Words &out, T value) { out.push_back(static_cast<std::uint64_t>(value)); }
using CardWords = std::array<std::uint64_t, 8>;
CardWords cardWords(const CardInstance &c) {
    return {static_cast<std::uint64_t>(c.id), static_cast<std::uint64_t>(c.upgraded),
            static_cast<std::uint64_t>(c.specialData), static_cast<std::uint64_t>(c.cost),
            static_cast<std::uint64_t>(c.costForTurn), static_cast<std::uint64_t>(c.freeToPlayOnce),
            static_cast<std::uint64_t>(c.retain), static_cast<std::uint64_t>(c.uniqueId)};
}
void card(Words &out, const CardInstance &c, bool identity) {
    const auto words = cardWords(c);
    out.insert(out.end(), words.begin(), words.begin() + (identity ? 8 : 7));
}
bool cardLess(const CardInstance &a, const CardInstance &b) {
    return std::tie(a.id, a.upgraded, a.specialData, a.cost, a.costForTurn,
                    a.freeToPlayOnce, a.retain, a.uniqueId)
         < std::tie(b.id, b.upgraded, b.specialData, b.cost, b.costForTurn,
                    b.freeToPlayOnce, b.retain, b.uniqueId);
}
template<class Iterator> void pile(Words &out, Iterator first, Iterator last,
                                    bool unordered, bool identity) {
    add(out, last - first);
    if (unordered) {
        std::vector<CardWords> cards;
        for (; first != last; ++first) {
            auto words = cardWords(*first);
            if (!identity) words.back() = 0;
            cards.push_back(words);
        }
        std::sort(cards.begin(), cards.end());
        for (const auto &words : cards)
            out.insert(out.end(), words.begin(), words.begin() + (identity ? 8 : 7));
    } else for (; first != last; ++first) card(out, *first, identity);
}
void player(Words &out, const Player &p) {
    add(out, p.cc);
    add(out, p.gold);
    add(out, p.curHp);
    add(out, p.maxHp);
    add(out, p.energy);
    add(out, p.energyPerTurn);
    add(out, p.cardDrawPerTurn);
    add(out, p.stance);
    add(out, p.orbSlots);
    add(out, p.lightningOrbsChanneled);
    add(out, p.frostOrbsChanneled);
    add(out, p.lastTargetedMonster);
    add(out, p.block);
    add(out, p.artifact);
    add(out, p.dexterity);
    add(out, p.focus);
    add(out, p.strength);
    add(out, p.justAppliedBits);
    add(out, p.statusBits0);
    add(out, p.statusBits1);
    add(out, p.relicBits0);
    add(out, p.relicBits1);
    add(out, p.happyFlowerCounter);
    add(out, p.incenseBurnerCounter);
    add(out, p.inkBottleCounter);
    add(out, p.inserterCounter);
    add(out, p.nunchakuCounter);
    add(out, p.penNibCounter);
    add(out, p.sundialCounter);
    add(out, p.haveUsedNecronomiconThisTurn);
    add(out, p.combustHpLoss);
    add(out, p.devaFormEnergyPerTurn);
    add(out, p.echoFormCardsDoubled);
    add(out, p.panacheCounter);
    add(out, p.cardsPlayedThisTurn);
    add(out, p.attacksPlayedThisTurn);
    add(out, p.skillsPlayedThisTurn);
    add(out, p.cardsDiscardedThisTurn);
    add(out, p.lastAttackUnblockedDamage);
    add(out, p.timesDamagedThisCombat);
    add(out, p.bomb1);
    add(out, p.bomb2);
    add(out, p.bomb3);
    add(out, p.selfRepairHeal);
    add(out, p.creativeAi);
    add(out, p.helloWorld);
    add(out, p.heatsinks);
    add(out, p.nirvanaBlock);
    add(out, p.rushdownDraw);
    add(out, p.studyInsights);
    add(out, p.mentalFortressBlock);
    add(out, p.machineLearning);
    add(out, p.storm);
    add(out, p.orangePelletsCardTypesPlayed.to_ulong());
    for (int i = 0; i < p.orbSlots; ++i) { add(out, p.orbs[i]); add(out, p.orbData[i]); }
    add(out, p.statusMap.size());
    for (const auto &entry : p.statusMap) { add(out, entry.first); add(out, entry.second); }
}
void monster(Words &out, const Monster &m, bool hidden, bool dome) {
    add(out, m.idx);
    add(out, m.id);
    add(out, m.curHp);
    add(out, m.maxHp);
    add(out, m.block);
    add(out, m.isEscapingB);
    add(out, m.halfDead);
    add(out, m.escapeNext);
    add(out, m.statusBits);
    add(out, m.artifact);
    add(out, m.blockReturn);
    add(out, m.choked);
    add(out, m.corpseExplosion);
    add(out, m.lockOn);
    add(out, m.mark);
    add(out, m.metallicize);
    add(out, m.platedArmor);
    add(out, m.poison);
    add(out, m.regen);
    add(out, m.shackled);
    add(out, m.strength);
    add(out, m.vulnerable);
    add(out, m.weak);
    add(out, m.uniquePower0);
    add(out, m.uniquePower1);
    if (hidden || !dome) add(out, m.moveHistory[0]);
    add(out, m.moveHistory[1]);
    if (hidden) add(out, m.miscInfo);
}
void rng(Words &out, const Random &r) {
    add(out, r.counter); add(out, r.seed0); add(out, r.seed1);
}
const CardInstance *selectionCard(const BattleContext &s, int index) {
    switch (s.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::HEADBUTT: case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION: case CardSelectTask::MEDITATE:
            return &s.cards.discardPile.at(index);
        case CardSelectTask::EXHUME: return &s.cards.exhaustPile.at(index);
        case CardSelectTask::SECRET_TECHNIQUE: case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK: case CardSelectTask::OMNISCIENCE:
            return &s.cards.drawPile.at(index);
        case CardSelectTask::ARMAMENTS: case CardSelectTask::DUAL_WIELD:
        case CardSelectTask::EXHAUST_ONE: case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE: case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP: case CardSelectTask::WARCRY:
        case CardSelectTask::DISCARD_ONE:
            return &s.cards.hand.at(index);
        default: return nullptr;
    }
}
}

std::uint64_t hashSearchWords(const Words &words) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const auto word : words) {
        // One mixed word per field; vector equality, not this hash, certifies fusion.
        auto x = word + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        hash = (hash ^ (x ^ (x >> 31))) * 0x100000001b3ULL;
    }
    return hash;
}

bool isSettledSearchState(const BattleContext &s) {
    return s.inputState == InputState::PLAYER_NORMAL && s.actionQueue.size == 0
        && s.cardQueue.size == 0 && !s.endTurnQueued && !s.endTurnAfterCurrentCard;
}

void canonicalizeSearchState(BattleContext &s) {
    if (!isSettledSearchState(s)) return;
    std::sort(s.cards.hand.begin(), s.cards.hand.begin() + s.cards.cardsInHand, cardLess);
    std::sort(s.cards.discardPile.begin(), s.cards.discardPile.end(), cardLess);
    std::sort(s.cards.exhaustPile.begin(), s.cards.exhaustPile.end(), cardLess);
}

Words publicSearchWords(const BattleContext &s) {
    Words out;
    out.reserve(256);
    add(out, s.turn); add(out, s.inputState); add(out, s.outcome);
    add(out, s.escapedCombat); add(out, s.unsupportedEffectKind); add(out, s.unsupportedEffectId);
    player(out, s.player);
    add(out, s.potionCapacity);
    for (int i = 0; i < s.potionCapacity; ++i) add(out, s.potions[i]);
    const auto &c = s.cards;
    pile(out, c.hand.begin(), c.hand.begin() + c.cardsInHand, true, false);
    pile(out, c.discardPile.begin(), c.discardPile.end(), true, false);
    pile(out, c.exhaustPile.begin(), c.exhaustPile.end(), true, false);
    pile(out, c.drawPile.begin(), c.drawPile.end(),
         !s.player.hasRelic<RelicId::FROZEN_EYE>(), false);
    add(out, s.monsters.monstersAlive); add(out, s.monsters.monsterCount);
    for (int i = 0; i < s.monsters.monsterCount; ++i) {
        const auto &m = s.monsters.arr[i];
        monster(out, m, false, s.player.hasRelic<RelicId::RUNIC_DOME>());
        if (!s.player.hasRelic<RelicId::RUNIC_DOME>() && m.isTargetable()) {
            const auto intent = m.getMoveBaseDamage(s);
            add(out, intent.damage); add(out, intent.attackCount);
        }
    }
    if (s.inputState == InputState::CARD_SELECT) {
        const auto &sel = s.cardSelectInfo;
        add(out, sel.cardSelectTask); add(out, sel.pickCount);
        add(out, sel.canPickZero); add(out, sel.canPickAnyNumber); add(out, sel.data0);
        if (sel.cardSelectTask == CardSelectTask::SCRY)
            pile(out, c.drawPile.begin(), c.drawPile.begin() + std::min(sel.pickCount, static_cast<int>(c.drawPile.size())), false, false);
        if (sel.cardSelectTask == CardSelectTask::DISCOVERY || sel.cardSelectTask == CardSelectTask::CODEX)
            for (const auto id : sel.cards) add(out, id);
    }
    return out;
}

std::uint64_t public_search_key(const BattleContext &s) {
    return hashSearchWords(publicSearchWords(s));
}

Words searchStateWords(const BattleContext &s) {
    if (!isSettledSearchState(s)) throw std::logic_error("fingerprint requires an empty-queue player decision");
    const auto &cards = s.cards;
    if (!std::is_sorted(cards.hand.begin(), cards.hand.begin() + cards.cardsInHand, cardLess)
        || !std::is_sorted(cards.discardPile.begin(), cards.discardPile.end(), cardLess)
        || !std::is_sorted(cards.exhaustPile.begin(), cards.exhaustPile.end(), cardLess))
        throw std::logic_error("fingerprint requires canonical hand/discard/exhaust piles");
    Words out;
    out.reserve(512);
    add(out, s.seed); add(out, s.floorNum); add(out, s.encounter); add(out, s.ascension);
    add(out, s.loopCount); add(out, s.energyWasted); add(out, s.cardsDrawn);
    add(out, s.haveUsedDiscoveryAction); add(out, s.undefinedBehaviorEvoked);
    rng(out, s.aiRng); rng(out, s.cardRandomRng); rng(out, s.miscRng);
    rng(out, s.monsterHpRng); rng(out, s.potionRng); rng(out, s.shuffleRng);
    add(out, s.outcome); add(out, s.inputState);
    add(out, s.unsupportedEffectKind); add(out, s.unsupportedEffectId);
    add(out, s.monsterTurnIdx); add(out, s.isBattleOver); add(out, s.escapedCombat);
    add(out, s.endTurnQueued); add(out, s.endTurnAfterCurrentCard); add(out, s.turnHasEnded);
    add(out, s.skipMonsterTurn); add(out, s.turn); add(out, s.miscBits.to_ulong());
    add(out, s.potionCount); add(out, s.potionCapacity);
    for (int i = 0; i < s.potionCapacity; ++i) add(out, s.potions[i]);
    player(out, s.player);
    const auto &c = s.cards;
    add(out, c.nextUniqueCardId); add(out, c.handNormalityCount); add(out, c.handPainCount);
    add(out, c.strikeCount); add(out, c.handBloodCardCount);
    add(out, c.drawPileBloodCardCount); add(out, c.discardPileBloodCardCount);
    pile(out, c.hand.begin(), c.hand.begin() + c.cardsInHand, false, true);
    pile(out, c.discardPile.begin(), c.discardPile.end(), false, true);
    pile(out, c.exhaustPile.begin(), c.exhaustPile.end(), false, true);
    pile(out, c.drawPile.begin(), c.drawPile.end(), false, true);
    for (const auto &stasis : c.stasisCards) card(out, stasis, true);
    pile(out, s.nightmareCards.begin(), s.nightmareCards.end(), false, true);
    add(out, s.nightmareCopyCounts.size());
    for (const auto count : s.nightmareCopyCounts) add(out, count);
    add(out, s.monsters.monstersAlive); add(out, s.monsters.monsterCount);
    add(out, s.monsters.extraRollMoveOnTurn.to_ulong()); add(out, s.monsters.skipTurn.to_ulong());
    for (int i = 0; i < s.monsters.monsterCount; ++i) monster(out, s.monsters.arr[i], true, false);
    // curCardQueueItem, CardSelectInfo, limbo and unused queue slots are stale here:
    // they are overwritten before the next use.
    return out;
}

std::uint64_t semantic_action_key(const BattleContext &s, Action a) {
    Words out;
    add(out, a.getActionType());
    if (s.inputState == InputState::PLAYER_NORMAL && a.getActionType() == ActionType::CARD) {
        card(out, s.cards.hand[a.getSourceIdx()], false); add(out, a.getTargetIdx());
    } else if (s.inputState == InputState::PLAYER_NORMAL && a.getActionType() == ActionType::POTION) {
        add(out, s.potions[a.getSourceIdx()]); add(out, a.getTargetIdx());
    } else if (s.inputState == InputState::CARD_SELECT && a.getActionType() == ActionType::SINGLE_CARD_SELECT) {
        if (const auto *c = selectionCard(s, a.getSelectIdx())) card(out, *c, false);
        else add(out, a.bits);
    } else if (s.inputState == InputState::CARD_SELECT && a.getActionType() == ActionType::MULTI_CARD_SELECT
               && s.cardSelectInfo.cardSelectTask != CardSelectTask::SCRY) {
        std::vector<CardWords> selected;
        for (const auto i : a.getSelectedIdxs()) {
            auto words = cardWords(s.cards.hand[i]); words.back() = 0; selected.push_back(words);
        }
        std::sort(selected.begin(), selected.end()); add(out, selected.size());
        for (const auto &words : selected) out.insert(out.end(), words.begin(), words.begin() + 7);
    } else add(out, a.bits);  // Scry positions and non-card choices retain their meaning.
    return hashSearchWords(out);
}

Action mapSemanticAction(const BattleContext &source, Action action, const BattleContext &target) {
    // Normal decisions have a small indexed hand; do not enumerate an entire legal
    // menu just to relocate one card. Positional Scry and choice screens stay separate.
    if (source.inputState == InputState::PLAYER_NORMAL && target.inputState == InputState::PLAYER_NORMAL) {
        const auto type = action.getActionType();
        if (type == ActionType::CARD) {
            const auto words = cardWords(source.cards.hand[action.getSourceIdx()]);
            for (int i = 0; i < target.cards.cardsInHand; ++i) {
                const auto other = cardWords(target.cards.hand[i]);
                if (!std::equal(words.begin(), words.begin() + 7, other.begin())) continue;
                const Action mapped(type, i, action.getTargetIdx());
                if (mapped.isValidAction(target)) return mapped;
            }
        } else if (type == ActionType::POTION) {
            for (int i = 0; i < target.potionCapacity; ++i) {
                if (source.potions[action.getSourceIdx()] != target.potions[i]) continue;
                const Action mapped(type, i, action.getTargetIdx());
                if (mapped.isValidAction(target)) return mapped;
            }
        } else if (action.isValidAction(target)) return action;
        throw std::runtime_error("normal public action has no legal semantic match");
    }
    const auto key = semantic_action_key(source, action);
    for (const auto candidate : Action::getAllActionsInState(target))
        if (candidate.isValidAction(target) && semantic_action_key(target, candidate) == key) return candidate;
    throw std::runtime_error("public action has no legal semantic match");
}
}
