//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_ML_CARDQUEUE_H
#define STS_ML_CARDQUEUE_H

#include <array>
#include <cassert>

#include "combat/CardInstance.h"

namespace sts {

    struct CardQueueItem {
        CardInstance card; // todo taking a copy is is incorrect because the card needs to be read from hand I believe in some scenarios, not sure
//        int cardIdx;
        int target = 0;

        // special data
        bool isEndTurn = false;
        bool triggerOnUse = true;
        bool ignoreEnergyTotal = false;
        int energyOnUse = 0;
        bool freeToPlay = false;
        bool randomTarget = false;
        bool autoplay = false;
        int regretCardCount = 0; // maybe use to indicate triggerOnUse too?

        bool purgeOnUse = false;
        bool exhaustOnUse = false;

        // Every field, in a fixed order, for code that stores states: see combat/StateFields.h.
        template <class Self, class Visit> static void visitFields(Self &s, Visit &&visit) {
            visit(s.card); visit(s.target); visit(s.isEndTurn); visit(s.triggerOnUse); visit(s.ignoreEnergyTotal);
            visit(s.energyOnUse); visit(s.freeToPlay); visit(s.randomTarget); visit(s.autoplay);
            visit(s.regretCardCount); visit(s.purgeOnUse); visit(s.exhaustOnUse);
        }

        CardQueueItem() = default;
        CardQueueItem(const CardInstance &card, int target, int energyOnUse) : card(card), target(target), energyOnUse(energyOnUse) {}

        static CardQueueItem endTurnItem() {
            CardQueueItem ret;
            ret.isEndTurn = true;
            return ret;
        }
    };

    struct CardQueue {
        static constexpr int capacity = 10;
        int size = 0;
        int backIdx = 0;
        int frontIdx = 0;
        std::array<CardQueueItem, capacity> arr;

        // Only the queued items: the slots outside the ring keep whatever was popped from them.
        template <class Self, class Visit> static void visitFields(Self &s, Visit &&visit) {
            visit(s.size); visit(s.backIdx); visit(s.frontIdx);
            for (int i = 0; i < s.size; ++i) {
                visit(s.arr.at((s.frontIdx + i) % capacity));
            }
        }

        void clear();
        void pushFront(CardQueueItem item);
        void pushBack(CardQueueItem item);
        [[nodiscard]] bool isEmpty() const;
        [[nodiscard]] bool containsCardWithId(int uniqueId) const;

        CardQueueItem popFront();
        CardQueueItem popBack();
        CardQueueItem &front();


    };

}


#endif //STS_ML_CARDQUEUE_H
