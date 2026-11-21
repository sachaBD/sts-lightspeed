//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_ML_ACTIONQUEUE_H
#define STS_ML_ACTIONQUEUE_H

#include "sts_common.h"
#include "combat/QueuedAction.h"
#include <bitset>
#include <cassert>


namespace sts {

    // Simple deque
    template<int capacity>
    struct ActionQueue {
        friend BattleContext;
        int front = 0;
        int back = 0;
        int size = 0;
        std::bitset<capacity> bits; // for the shouldClear field

        std::array<ActionPayload, capacity> arr;

        template <class Self, class Visit> static void visitFields(Self& s, Visit&& visit) {
            visit(s.front); visit(s.back); visit(s.size); visit(s.bits);
            for (int i = 0; i < s.size; ++i) visit(s.arr.at((s.front + i) % capacity));
        }

        void clear();
        void pushFront(Action a);
        void pushBack(Action a);
        bool isEmpty();
        Action popFront();
        [[nodiscard]] int getCapacity() const;
    };

    template<int capacity>
    void ActionQueue<capacity>::clear() {
        // Release payloads in inactive slots as well.
        for (auto &effect : arr) effect = queued::NoOp{};
        size = 0;
        back = 0;
        front = 0;
    }

    template <int capacity>
    void ActionQueue<capacity>::pushFront(Action a) {
#ifdef sts_asserts
        assert(size != capacity);
#endif
        --front;
        ++size;
        if (front < 0) {
            front = capacity-1;
        }
        bits.set(front, a.clearOnCombatVictory);
        arr[front] = std::move(a.payload);
    }

    template<int capacity>
    void ActionQueue<capacity>::pushBack(Action a) {
#ifdef sts_asserts
        if (size >= capacity) {
            assert(false);
        }
#endif
        if (back >= capacity) {
            back = 0;
        }
        bits.set(back, a.clearOnCombatVictory);
        arr[back] = std::move(a.payload);
        ++back;
        ++size;
    }

    template<int capacity>
    bool ActionQueue<capacity>::isEmpty() {
        return size == 0;
    }

    template<int capacity>
    Action ActionQueue<capacity>::popFront() {
#ifdef sts_asserts
        assert(size > 0 );
#endif
        // Inactive slots must not retain captured cards or vectors.
        Action a{std::move(arr[front]), bits[front]};
        arr[front] = queued::NoOp{};
        ++front;
        --size;
        if (front >= capacity) {
            front = 0;
        }
        return a;
    }

    template<int capacity>
    int ActionQueue<capacity>::getCapacity() const {
        return capacity;
    }

}


#endif //STS_ML_ACTIONQUEUE_H
