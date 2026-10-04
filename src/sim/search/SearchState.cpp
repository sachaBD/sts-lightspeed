#include "sim/search/SearchState.h"
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace sts::search {
std::uint64_t publicMonsterHistory(const Monster &m) {
    switch (m.id) {
        // These flags are set by the visible Haste/Rebirth actions.
        case MonsterId::TIME_EATER: case MonsterId::AWAKENED_ONE:
            return m.miscInfo != 0;
        // Low bits count announced Defensive Stances; bit 2 is announced Anger.
        case MonsterId::THE_CHAMP:
            return static_cast<std::uint64_t>(m.miscInfo & 7);
        default: return 0; // never expose arbitrary private monster scratch data
    }
}
namespace {
using Words = std::vector<std::uint64_t>;
template<class T> void add(Words &out, T value) { out.push_back(static_cast<std::uint64_t>(value)); }
// Injective packing of all public per-instance card attributes. Identity/slot is
// deliberately excluded; multiplicities remain explicit in each sorted multiset.
std::uint64_t cardWord(const CardInstance &c) {
    return static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.id))
        | static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.specialData)) << 16
        | static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.cost)) << 32
        | static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.costForTurn)) << 40
        | static_cast<std::uint64_t>(c.upgraded) << 48
        | static_cast<std::uint64_t>(c.freeToPlayOnce) << 49
        | static_cast<std::uint64_t>(c.retain) << 50;
}
template<class Iterator> void pile(Words &out, Iterator first, Iterator last, bool unordered) {
    add(out, last - first);
    const auto begin = out.size();
    for (; first != last; ++first) add(out, cardWord(*first));
    if (unordered) std::sort(out.begin() + begin, out.end());
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
void monster(Words &out, const Monster &m, bool dome) {
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
    if (m.id == MonsterId::TIME_EATER || m.id == MonsterId::AWAKENED_ONE
        || m.id == MonsterId::THE_CHAMP) add(out, publicMonsterHistory(m));
    if (!dome) add(out, m.moveHistory[0]);
    add(out, m.moveHistory[1]);
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

namespace {
std::uint64_t hashWords(const std::uint64_t *first, const std::uint64_t *last) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (auto p = first; p != last; ++p) {
        const auto word = *p;
        // Hash indexes the public signature; vector equality checks accidental collisions.
        auto x = word + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        hash = (hash ^ (x ^ (x >> 31))) * 0x100000001b3ULL;
    }
    return hash;
}
}
std::uint64_t hashSearchWords(const Words &words) {
    return hashWords(words.data(), words.data() + words.size());
}

bool isSettledSearchState(const BattleContext &s) {
    return s.inputState == InputState::PLAYER_NORMAL && s.actionQueue.size == 0
        && s.cardQueue.size == 0 && !s.endTurnQueued && !s.endTurnAfterCurrentCard;
}

void publicSearchWords(const BattleContext &s, Words &out) {
    out.clear();
    out.reserve(256);
    add(out, s.turn); add(out, s.inputState); add(out, s.outcome);
    add(out, s.escapedCombat); add(out, s.unsupportedEffectKind); add(out, s.unsupportedEffectId);
    player(out, s.player);
    add(out, s.potionCapacity);
    for (int i = 0; i < s.potionCapacity; ++i) add(out, s.potions[i]);
    const auto &c = s.cards;
    pile(out, c.hand.begin(), c.hand.begin() + c.cardsInHand, true);
    pile(out, c.discardPile.begin(), c.discardPile.end(), true);
    pile(out, c.exhaustPile.begin(), c.exhaustPile.end(), true);
    const auto size = static_cast<int>(c.drawPile.size());
    if (c.knownDrawTop < 0 || c.knownDrawBottom < 0 || c.knownDrawTop + c.knownDrawBottom > size)
        throw std::logic_error("invalid public draw knowledge");
    pile(out, c.drawPile.begin(), c.drawPile.end(), !s.player.hasRelic<RelicId::FROZEN_EYE>());
    add(out, c.knownDrawBottom); add(out, c.knownDrawTop);
    pile(out, c.drawPile.begin(), c.drawPile.begin() + c.knownDrawBottom, false);
    pile(out, c.drawPile.end() - c.knownDrawTop, c.drawPile.end(), false);
    // Pending Nightmare copies and stasis cards are public persistent effects.
    for (const auto &card : c.stasisCards) add(out, cardWord(card));
    pile(out, s.nightmareCards.begin(), s.nightmareCards.end(), false);
    add(out, s.nightmareCopyCounts.size());
    for (const auto count : s.nightmareCopyCounts) add(out, count);
    add(out, s.monsters.monstersAlive); add(out, s.monsters.monsterCount);
    for (int i = 0; i < s.monsters.monsterCount; ++i) {
        const auto &m = s.monsters.arr[i];
        monster(out, m, s.player.hasRelic<RelicId::RUNIC_DOME>());
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
            pile(out, c.drawPile.begin(), c.drawPile.begin() + std::min(sel.pickCount, static_cast<int>(c.drawPile.size())), false);
        if (sel.cardSelectTask == CardSelectTask::DISCOVERY || sel.cardSelectTask == CardSelectTask::CODEX)
            for (const auto id : sel.cards) add(out, id);
    }
}

Words publicSearchWords(const BattleContext &s) {
    Words out;
    publicSearchWords(s, out);
    return out;
}

std::uint64_t public_search_key(const BattleContext &s) {
    return hashSearchWords(publicSearchWords(s));
}

std::uint64_t semantic_action_key(const BattleContext &s, Action a) {
    // Every representable action fits: type, selection count, at most ten cards.
    std::array<std::uint64_t, 12> fields{};
    std::size_t count = 1;
    fields[0] = static_cast<std::uint64_t>(a.getActionType());
    if (s.inputState == InputState::PLAYER_NORMAL && a.getActionType() == ActionType::CARD) {
        fields[count++] = cardWord(s.cards.hand[a.getSourceIdx()]); fields[count++] = a.getTargetIdx();
    } else if (s.inputState == InputState::PLAYER_NORMAL && a.getActionType() == ActionType::POTION) {
        fields[count++] = static_cast<std::uint64_t>(s.potions[a.getSourceIdx()]); fields[count++] = a.getTargetIdx();
    } else if (s.inputState == InputState::CARD_SELECT && a.getActionType() == ActionType::SINGLE_CARD_SELECT) {
        const auto *card = selectionCard(s, a.getSelectIdx());
        fields[count++] = card ? cardWord(*card) : a.bits;
    } else if (s.inputState == InputState::CARD_SELECT && a.getActionType() == ActionType::MULTI_CARD_SELECT
               && s.cardSelectInfo.cardSelectTask != CardSelectTask::SCRY) {
        const auto selected = a.getSelectedIdxs();
        fields[count++] = selected.size();
        for (const auto i : selected) fields[count++] = cardWord(s.cards.hand[i]);
        std::sort(fields.begin() + 2, fields.begin() + count);
    } else fields[count++] = a.bits;
    return hashWords(fields.data(), fields.data() + count);
}

Action mapSemanticAction(const BattleContext &source, Action action, const BattleContext &target) {
    // Normal decisions have a small indexed hand; do not enumerate an entire legal
    // menu just to relocate one card. Positional Scry and choice screens stay separate.
    if (source.inputState == InputState::PLAYER_NORMAL && target.inputState == InputState::PLAYER_NORMAL) {
        const auto type = action.getActionType();
        if (type == ActionType::CARD) {
            const auto word = cardWord(source.cards.hand[action.getSourceIdx()]);
            for (int i = 0; i < target.cards.cardsInHand; ++i) {
                if (word != cardWord(target.cards.hand[i])) continue;
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
