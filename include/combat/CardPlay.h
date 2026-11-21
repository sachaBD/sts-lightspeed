#pragma once
#include "combat/CardInstance.h"

namespace sts {
    struct CardPlay {
        CardInstance card;
        int target = 0;

        bool triggerOnUse = true;
        bool ignoreEnergyTotal = false;
        int energyOnUse = 0;
        bool freeToPlay = false;
        bool randomTarget = false;
        bool autoplay = false;
        int regretCardCount = 0;

        bool purgeOnUse = false;
        bool exhaustOnUse = false;

        // Every field, in a fixed order, for code that stores states: see combat/StateFields.h.
        template <class Self, class Visit> static void visitFields(Self &s, Visit &&visit) {
            visit(s.card); visit(s.target); visit(s.triggerOnUse); visit(s.ignoreEnergyTotal);
            visit(s.energyOnUse); visit(s.freeToPlay); visit(s.randomTarget); visit(s.autoplay);
            visit(s.regretCardCount); visit(s.purgeOnUse); visit(s.exhaustOnUse);
        }

        CardPlay() = default;
        CardPlay(const CardInstance &card, int target, int energyOnUse) : card(card), target(target), energyOnUse(energyOnUse) {}

    };
}
