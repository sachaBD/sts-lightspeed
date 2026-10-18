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
                resp = {{"state", state_json(*bc)}, {"actions", actions_json(*bc)}, {"ai_rolls", rolls_since(fresh, bc->aiRng)}};
            } else if (cmd == "step") {
                if (!bc) throw std::runtime_error{"no fight"};
                const search::Action a{r.at("bits").get<std::uint32_t>()};
                if (!a.isValidAction(*bc)) throw std::runtime_error{"invalid action"};
                const Random before = bc->aiRng;
                a.execute(*bc);
                resp = {{"state", state_json(*bc)}, {"actions", actions_json(*bc)}, {"ai_rolls", rolls_since(before, bc->aiRng)}};
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
