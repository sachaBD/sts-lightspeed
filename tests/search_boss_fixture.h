#pragma once
#include "game/GameContext.h"
#include "combat/BattleContext.h"

// Synthetic scaling/draw/exhaust deck: not a qualified release loadout.
inline sts::BattleContext bossFixture(sts::MonsterEncounter encounter, std::uint64_t seed) {
    using namespace sts;
    GameContext game(CharacterClass::IRONCLAD, seed, 20);
    game.act = encounter == MonsterEncounter::CHAMP ? 2 : 3;
    game.floorNum = game.act == 2 ? 34 : 51;
    game.curHp = game.maxHp = 100;
    game.deck = Deck{};
    const auto add = [&](CardId id, int count, bool upgraded = true) {
        for (int i = 0; i < count; ++i) game.deck.obtainRaw(Card(id, upgraded));
    };
    add(CardId::BASH,1); add(CardId::STRIKE_RED,2,false); add(CardId::DEFEND_RED,2,false);
    add(CardId::POMMEL_STRIKE,3); add(CardId::SHRUG_IT_OFF,2); add(CardId::BATTLE_TRANCE,2);
    add(CardId::HEADBUTT,2); add(CardId::TWIN_STRIKE,3); add(CardId::UPPERCUT,2);
    add(CardId::TRUE_GRIT,2); add(CardId::SHOCKWAVE,1); add(CardId::INFLAME,1);
    add(CardId::DEMON_FORM,1); add(CardId::LIMIT_BREAK,1); add(CardId::OFFERING,1);
    add(CardId::BURNING_PACT,2); add(CardId::FLAME_BARRIER,1); add(CardId::DISARM,1);
    add(CardId::IMPERVIOUS,1); add(CardId::WHIRLWIND,1);
    BattleContext root; root.init(game, encounter);
    root.player.energy = root.player.energyPerTurn = 4;
    return root;
}
