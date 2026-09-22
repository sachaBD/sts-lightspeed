//
// Created by keega on 9/17/2021.
//

#ifndef STS_ML_ACTION_H
#define STS_ML_ACTION_H

#include "sts_common.h"
#include "data_structure/fixed_list.h"

#include "combat/BattleContext.h"


#include <iostream>


namespace sts {
    struct BattleContext;
}


namespace sts::search {

    enum class ActionType {
        CARD=0,
        POTION,
        SINGLE_CARD_SELECT,
        MULTI_CARD_SELECT,
        END_TURN,
        CHOOSE_STANCE,
        CHOOSE_ENEMY,
    };

    // couldn't make a union work in only 32 bits
    struct Action {
//        ActionType actionType;
//        int idx1;
//        int idx2;
        std::uint32_t bits = -1;

        Action() = default;
        Action(std::uint32_t bits);
        Action(ActionType actionType);
        Action(ActionType actionType, int idx1);
        Action(ActionType actionType, int idx1, int idx2);

        bool operator==(const Action &rhs) const;
        bool operator!=(const Action &rhs) const;

        [[nodiscard]] ActionType getActionType() const;
        [[nodiscard]] int getSourceIdx() const; // for card/potion actions
        [[nodiscard]] int getTargetIdx() const; // for card/potion actions

        [[nodiscard]] int getSelectIdx() const;   // for single card select action
        [[nodiscard]] fixed_list<int,10> getSelectedIdxs() const;         // for multi card select actions

        [[nodiscard]] bool isValidAction(const sts::BattleContext &bc) const;
        std::ostream& printDesc(std::ostream &os, const sts::BattleContext &bc) const;
        void execute(BattleContext &bc) const;


        static std::vector<Action> enumerateCardSelectActions(const BattleContext &bc);
        // Enumerates every currently representable battle decision.  States
        // outside PLAYER_NORMAL and CARD_SELECT intentionally return empty;
        // callers must report those simulator gaps rather than treating them
        // as an episode terminal.
        static std::vector<Action> getAllActionsInState(const BattleContext &bc);
    };

}


#endif //STS_ML_ACTION_H
