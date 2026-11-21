//
// Created by gamerpuppy on 8/24/2021.
//

#ifndef STS_ML_CARDSELECTINFO_H
#define STS_ML_CARDSELECTINFO_H

#include <array>

#include "constants/Cards.h"

namespace sts {

    enum class CardSelectTask {
        INVALID=0,
        ARMAMENTS,
        CODEX,
        DISCOVERY,
        DUAL_WIELD,
        EXHAUST_ONE,
        EXHAUST_MANY,
        EXHUME,
        FORETHOUGHT,
        GAMBLE,
        HEADBUTT,
        HOLOGRAM,
        LIQUID_MEMORIES_POTION,
        MEDITATE,
        NIGHTMARE,
        RECYCLE,
        SECRET_TECHNIQUE,
        SECRET_WEAPON,
        SEEK,
        SETUP,
        WARCRY,
        DISCARD_ONE,
        OMNISCIENCE,
        SCRY,
        WISH,
    };

    static constexpr const char* cardSelectTaskStrings[] = {
            "INVALID",
            "ARMAMENTS",
            "CODEX",
            "DISCOVERY",
            "DUAL_WIELD",
            "EXHAUST_ONE",
            "EXHAUST_MANY",
            "EXHUME",
            "FORETHOUGHT",
            "GAMBLE",
            "HEADBUTT",
            "HOLOGRAM",
            "LIQUID_MEMORIES_POTION",
            "MEDITATE",
            "NIGHTMARE",
            "RECYCLE",
            "SECRET_TECHNIQUE",
            "SECRET_WEAPON",
            "SEEK",
            "SETUP",
            "WARCRY",
            "DISCARD_ONE",
            "OMNISCIENCE",
            "SCRY",
            "WISH",
    };

    struct CardSelectInfo {
        std::array<CardId, 3> cards{CardId::INVALID, CardId::INVALID, CardId::INVALID};
        bool canPickZero = false;
        bool canPickAnyNumber = false;
        int pickCount = 0;
        int data0 = 0;

        CardSelectTask cardSelectTask = CardSelectTask::INVALID;

        // Every field, in a fixed order, for code that stores states: see combat/StateFields.h.
        template <class Self, class Visit> static void visitFields(Self &s, Visit &&visit) {
            visit(s.cards); visit(s.canPickZero); visit(s.canPickAnyNumber); visit(s.pickCount); visit(s.data0);
            visit(s.cardSelectTask);
        }

        std::array<CardId, 3>& discovery_Cards() { return cards; }
        int& discovery_CopyCount() { return data0; }
        int& dualWield_CopyCount() { return data0; }
        std::array<CardId, 3>& codexCards() { return cards; }
    };

}

#endif //STS_ML_CARDSELECTINFO_H
