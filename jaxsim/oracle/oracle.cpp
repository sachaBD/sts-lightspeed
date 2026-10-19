// jaxsim_oracle: the C++ simulator as a JSON-lines process, for jaxsim parity tests.
//
// One request per stdin line, one response per stdout line. All enum values are raw C++ values; the Python
// side maps them by name through jaxsim's generated *_CPP tables.
//
//   {"cmd":"start","seed":S,"hp":H,"max_hp":M,"deck":[[card,upgraded],...],"relics":[...],"potions":[...],
//    "potion_capacity":2}                                  -> {"state":..., "actions":[...], "ai_rolls":[...]}
//   {"cmd":"step","bits":B}                                -> {"state":..., "actions":[...], "ai_rolls":[...]}
//   {"cmd":"cards"}                                        -> {"cards":[[id, upgraded, cost, type, exhausts, ethereal]]}
//
// ai_rolls: the aiRng.random(99) values consumed by the request (The Champ's move rolls), recovered by replaying
// a copy of aiRng. Errors: {"error":"..."}. Ascension 20, act 2 boss room, encounter CHAMP.
#include "combat/BattleContext.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "sim/search/Action.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <set>
#include <string>

using Json = nlohmann::json;
using namespace sts;

namespace {

Json card_json(const CardInstance& c) {
    return {static_cast<int>(c.id), c.upgraded ? c.getUpgradeCount() : 0, c.cost, c.costForTurn, c.specialData};
}

template <typename It>
Json pile_json(It begin, It end) {
    Json out = Json::array();
    for (auto it = begin; it != end; ++it) out.push_back(card_json(*it));
    return out;
}

Json state_json(const BattleContext& bc) {
    Json p_status = Json::array();
    for (int s = 1; s <= static_cast<int>(PlayerStatus::THE_BOMB); ++s) {  // INVALID = 0 ... THE_BOMB = last
        const auto ps = static_cast<PlayerStatus>(s);
        if (ps == PlayerStatus::THE_BOMB) continue;  // stored as bomb1..3 counters, not via getStatusRuntime
        if (!bc.player.hasStatusRuntime(ps) && s != static_cast<int>(PlayerStatus::STRENGTH)
            && s != static_cast<int>(PlayerStatus::DEXTERITY) && s != static_cast<int>(PlayerStatus::ARTIFACT)) continue;
        // flag statuses (Barricade) have no map entry: report 1
        const auto it = bc.player.statusMap.find(ps);
        const bool counted = it != bc.player.statusMap.end() || s == static_cast<int>(PlayerStatus::STRENGTH)
            || s == static_cast<int>(PlayerStatus::DEXTERITY) || s == static_cast<int>(PlayerStatus::ARTIFACT);
        if (const int v = counted ? bc.player.getStatusRuntime(ps) : 1; v != 0) p_status.push_back({s, v});
    }
    Json relics = Json::array();
    for (int r = 0; r < 128; ++r)  // relic bits cover battle relics only (ids < 128)
        if (bc.player.hasRelicRuntime(static_cast<RelicId>(r))) relics.push_back(r);
    Json potions = Json::array();
    for (int i = 0; i < bc.potionCapacity; ++i) potions.push_back(static_cast<int>(bc.potions[i]));

    const auto& m = bc.monsters.arr[0];
    Json e_status = Json::array();
    for (int s = 0; s < static_cast<int>(MonsterStatus::INVALID); ++s)
        if (const int v = m.getStatusInternal(static_cast<MonsterStatus>(s)); v != 0) e_status.push_back({s, v});

    const auto& cm = bc.cards;
    return {
        {"input_state", static_cast<int>(bc.inputState)},
        {"outcome", static_cast<int>(bc.outcome)},
        {"turn", bc.turn},
        {"player", {{"hp", bc.player.curHp}, {"max_hp", bc.player.maxHp}, {"block", bc.player.block},
                    {"energy", bc.player.energy}, {"energy_per_turn", bc.player.energyPerTurn}, {"status", p_status}}},
        {"relics", relics},
        {"potions", potions},
        {"potion_capacity", bc.potionCapacity},
        {"hand", pile_json(cm.hand.begin(), cm.hand.begin() + cm.cardsInHand)},
        {"draw", pile_json(cm.drawPile.begin(), cm.drawPile.end())},  // back = top
        {"discard", pile_json(cm.discardPile.begin(), cm.discardPile.end())},
        {"exhaust", pile_json(cm.exhaustPile.begin(), cm.exhaustPile.end())},
        {"enemy", {{"hp", m.curHp}, {"max_hp", m.maxHp}, {"block", m.block},
                   {"move", static_cast<int>(m.moveHistory[0])}, {"last", static_cast<int>(m.moveHistory[1])},
                   {"misc", m.miscInfo}, {"status", e_status}}},
        {"select_task", static_cast<int>(bc.cardSelectInfo.cardSelectTask)},
    };
}

// The pile a single card select picks from (by task); nullptr for selects not over a pile.
const CardInstance* select_source(const BattleContext& bc, int idx) {
    const auto& cm = bc.cards;
    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::ARMAMENTS: case CardSelectTask::EXHAUST_ONE: case CardSelectTask::WARCRY:
        case CardSelectTask::DUAL_WIELD: case CardSelectTask::DISCARD_ONE:
            return idx < cm.cardsInHand ? &cm.hand[idx] : nullptr;
        case CardSelectTask::HEADBUTT: case CardSelectTask::LIQUID_MEMORIES_POTION:
            return idx < static_cast<int>(cm.discardPile.size()) ? &cm.discardPile[idx] : nullptr;
        case CardSelectTask::EXHUME:
            return idx < static_cast<int>(cm.exhaustPile.size()) ? &cm.exhaustPile[idx] : nullptr;
        default:
            return nullptr;
    }
}

std::set<int> uids(const BattleContext& bc, int pile) {  // 0 draw, 1 hand, 2 discard, 3 exhaust
    std::set<int> out;
    const auto& cm = bc.cards;
    if (pile == 1) for (int i = 0; i < cm.cardsInHand; ++i) out.insert(cm.hand[i].uniqueId);
    if (pile == 0) for (const auto& c : cm.drawPile) out.insert(c.uniqueId);
    if (pile == 2) for (const auto& c : cm.discardPile) out.insert(c.uniqueId);
    if (pile == 3) for (const auto& c : cm.exhaustPile) out.insert(c.uniqueId);
    return out;
}

// Cards drawn by a request: in hand now and in the draw or discard pile before. After END_TURN the old hand was
// discarded first, so any card that existed before counts.
// `left_hand`: a card that left the hand during the request (the played card, a Warcry pick); if it is back in
// hand it was drawn back (e.g. reshuffled and drawn by Dark Embrace).
Json drawn_json(const std::set<int>& draw_before, const std::set<int>& disc_before, const std::set<int>& hand_before,
                bool end_turn, int left_hand, int limbo, const BattleContext& bc) {
    Json out = Json::array();
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto& c = bc.cards.hand[i];
        const bool was_pile = draw_before.count(c.uniqueId) || disc_before.count(c.uniqueId);
        if (was_pile || (end_turn && hand_before.count(c.uniqueId)) || c.uniqueId == left_hand || c.uniqueId == limbo) out.push_back(card_json(c));
    }
    return out;
}

// Cards moved from hand to exhaust by a request (True Grit's random exhaust, choices).
Json exhausted_json(const std::set<int>& hand_before, const std::set<int>& exh_before, const BattleContext& bc) {
    Json out = Json::array();
    for (const auto& c : bc.cards.exhaustPile)
        if (hand_before.count(c.uniqueId) && !exh_before.count(c.uniqueId)) out.push_back(card_json(c));
    return out;
}

Json actions_json(const BattleContext& bc) {
    Json out = Json::array();
    for (const auto& a : search::Action::getAllActionsInState(bc)) {
        Json j = {{"bits", a.bits}, {"type", static_cast<int>(a.getActionType())}};
        switch (a.getActionType()) {
            case search::ActionType::CARD:
                j["hand_idx"] = a.getSourceIdx();
                j["card"] = card_json(bc.cards.hand[a.getSourceIdx()]);
                break;
            case search::ActionType::POTION:
                j["potion_idx"] = a.getSourceIdx();
                j["potion"] = static_cast<int>(bc.potions[a.getSourceIdx()]);
                j["discard"] = a.getTargetIdx() > 5;
                break;
            case search::ActionType::SINGLE_CARD_SELECT:
                j["select_idx"] = a.getSelectIdx();
                j["task"] = static_cast<int>(bc.cardSelectInfo.cardSelectTask);
                if (const auto* c = select_source(bc, a.getSelectIdx())) j["card"] = card_json(*c);
                break;
            default:
                break;
        }
        out.push_back(j);
    }
    return out;
}

Json rolls_since(Random before, const Random& after) {
    Json out = Json::array();
    while (before.counter < after.counter) out.push_back(before.random(99));
    return out;
}

GameContext make_game(const Json& r) {
    const std::uint64_t seed = r.at("seed").get<std::uint64_t>();
    GameContext gc{CharacterClass::IRONCLAD, seed, 20};
    gc.act = 2;
    gc.floorNum = 33;
    gc.info.encounter = MonsterEncounter::CHAMP;
    gc.curRoom = Room::BOSS;
    gc.curMapNodeX = gc.curMapNodeY = -7;  // never the burning elite
    gc.curHp = r.at("hp").get<int>();
    gc.maxHp = r.at("max_hp").get<int>();
    gc.potionCapacity = r.value("potion_capacity", 2);
    std::fill(gc.potions.begin(), gc.potions.end(), Potion::EMPTY_POTION_SLOT);
    gc.potionCount = 0;
    for (const auto& p : r.value("potions", Json::array())) gc.potions[gc.potionCount++] = static_cast<Potion>(p.get<int>());
    gc.relics = RelicContainer{};
    for (const auto& id : r.value("relics", Json::array())) gc.relics.add({static_cast<RelicId>(id.get<int>()), 0});
    gc.deck = Deck{};
    for (const auto& c : r.at("deck")) {
        Card card{static_cast<CardId>(c.at(0).get<int>())};
        for (int i = 0; i < c.at(1).get<int>(); ++i) card.upgrade();
        gc.deck.obtainRaw(card);
    }
    gc.screenState = ScreenState::BATTLE;
    return gc;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::optional<BattleContext> bc;
    std::string line;
    while (std::getline(std::cin, line)) {
        Json resp;
        try {
            const auto r = Json::parse(line);
            const auto cmd = r.at("cmd").get<std::string>();
            if (cmd == "start") {
                const auto gc = make_game(r);
                bc.emplace();
                bc->init(gc);
                const Random fresh(gc.seed + gc.floorNum);  // BattleContext::init's aiRng seed
                Json drawn = Json::array();
                for (int i = 0; i < bc->cards.cardsInHand; ++i) drawn.push_back(card_json(bc->cards.hand[i]));
                resp = {{"state", state_json(*bc)}, {"actions", actions_json(*bc)}, {"ai_rolls", rolls_since(fresh, bc->aiRng)},
                        {"drawn", drawn}, {"exhausted", Json::array()}};
            } else if (cmd == "step") {
                if (!bc) throw std::runtime_error{"no fight"};
                const search::Action a{r.at("bits").get<std::uint32_t>()};
                if (!a.isValidAction(*bc)) throw std::runtime_error{"invalid action"};
                const Random before = bc->aiRng;
                const auto draw_b = uids(*bc, 0), hand_b = uids(*bc, 1), disc_b = uids(*bc, 2), exh_b = uids(*bc, 3);
                int left_hand = -1;
                if (a.getActionType() == search::ActionType::CARD) left_hand = bc->cards.hand[a.getSourceIdx()].uniqueId;
                if (a.getActionType() == search::ActionType::SINGLE_CARD_SELECT
                    && bc->cardSelectInfo.cardSelectTask == CardSelectTask::WARCRY) left_hand = bc->cards.hand[a.getSelectIdx()].uniqueId;
                // the card being played while a choice is pending (in limbo; it may be reshuffled and drawn back)
                const int limbo = a.getActionType() == search::ActionType::SINGLE_CARD_SELECT
                    ? bc->curCardQueueItem.card.uniqueId : -1;
                a.execute(*bc);
                resp = {{"state", state_json(*bc)}, {"actions", actions_json(*bc)}, {"ai_rolls", rolls_since(before, bc->aiRng)},
                        {"drawn", drawn_json(draw_b, disc_b, hand_b, a.getActionType() == search::ActionType::END_TURN, left_hand, limbo, *bc)}, {"exhausted", exhausted_json(hand_b, exh_b, *bc)}};
            } else if (cmd == "cards") {
                Json cards = Json::array();
                for (int id = 1; id < static_cast<int>(CardId::ZAP) + 1; ++id) {
                    for (int up = 0; up < 2; ++up) {
                        const CardInstance c{static_cast<CardId>(id), up == 1};
                        cards.push_back({id, up, c.cost, static_cast<int>(c.getType()), c.doesExhaust(), c.isEthereal()});
                    }
                }
                resp = {{"cards", cards}};
            } else {
                throw std::runtime_error{"unknown cmd " + cmd};
            }
        } catch (const std::exception& e) {
            resp = {{"error", e.what()}};
        }
        std::cout << resp.dump() << '\n' << std::flush;
    }
}
