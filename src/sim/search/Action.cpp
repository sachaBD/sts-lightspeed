//
// Created by keega on 9/17/2021.
//

#include "sim/search/Action.h"

#include <functional>

namespace sts {

search::Action::Action(std::uint32_t bits) : bits(bits) {}
search::Action::Action(search::ActionType actionType) : bits(static_cast<int>(actionType) << 29) {}
search::Action::Action(search::ActionType actionType, int idx1) : bits((static_cast<int>(actionType) << 29) | (idx1 & 0xFFFF)) {}
search::Action::Action(search::ActionType actionType, int idx1, int idx2) : bits((static_cast<int>(actionType) << 29) | ((idx2 & 0x1FFF) << 16) | (idx1 & 0xFFFF)) {}
//
//search::Action::Action(search::ActionType actionType) : Action(actionType, 0, 0) {}
//
//search::Action::Action(search::ActionType actionType, int idx1) : Action(actionType, idx1, 0) {}
//
//search::Action::Action(search::ActionType actionType, int idx1, int idx2) : actionType(actionType), idx1(idx1),
//                                                                            idx2(idx2) {}

bool search::Action::operator==(const search::Action &rhs) const {
    return bits == rhs.bits;
//    return actionType == rhs.actionType && idx1 == rhs.idx1 && idx2 == rhs.idx2;
}

bool search::Action::operator!=(const search::Action &rhs) const {
    return !(rhs == *this);
}

search::ActionType search::Action::getActionType() const {
    return static_cast<ActionType>((bits >> 29) & 0xF);
//    return actionType;
}

int search::Action::getSourceIdx() const {
//    return idx1;
    return bits & 0xFFFF;
}

int search::Action::getTargetIdx() const {
//    return idx2;
    return (bits >> 16) & 0x1FFF;
}

int search::Action::getSelectIdx() const {
//    return idx1;
    return bits & 0xFFFF;
}

fixed_list<int, 10> search::Action::getSelectedIdxs() const {
    fixed_list<int,10> ret;
    int i = 0;
    int bitsCopy = bits & 0x3FF;
    while (bitsCopy) {
        if (bitsCopy & 0x1) {
            ret.push_back(i);
        }
        bitsCopy >>= 1;
        ++i;
    }
    return ret;
}

bool isValidPotionAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::PLAYER_NORMAL) {
        return false;
    }

    if (a.getSourceIdx() < 0 || a.getSourceIdx() >= 5) {
        return false;
    }

    const auto p = bc.potions[a.getSourceIdx()];
    if (p == Potion::INVALID || p == Potion::EMPTY_POTION_SLOT) {
        return false;
    }

    if (a.getTargetIdx() > 5) {
        // discard action
        return true;

    } else {

        if (p == sts::Potion::FAIRY_POTION) {
            return false;
        }

        // The base game disables Smoke Bomb in boss encounters. Allowing it
        // here both changes gameplay and strands the run beyond the final map
        // row when an agent attempts to escape a boss.
        if (p == Potion::SMOKE_BOMB && isBossEncounter(bc.encounter)) {
            return false;
        }

        if (!potionRequiresTarget(p)) {
            return true;
        }

        if (a.getTargetIdx() < 0) {
            return false;
        }

        return bc.monsters.arr[a.getTargetIdx()].isTargetable();
    }


}

bool isValidCardAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::PLAYER_NORMAL) {
        return false;
    }

    if (!bc.isCardPlayAllowed()) {
        return false;
    }

    if (a.getSourceIdx() < 0 || a.getSourceIdx() >= bc.cards.cardsInHand) {
        return false;
    }

    const auto &c = bc.cards.hand[a.getSourceIdx()];

    if (a.getTargetIdx() < 0 || a.getTargetIdx() >= 5) {
        return false;
    }

    return c.canUse(bc, a.getTargetIdx(), false);
}

bool isValidSingleCardSelectAction(const BattleContext &bc, const search::Action &a) {
    if (bc.inputState != InputState::CARD_SELECT) {
        return false;
    }

    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::CODEX:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < 4;

        case CardSelectTask::DISCOVERY:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < 3;

        case CardSelectTask::EXHUME:
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.exhaustPile.size()) {
                return false;
            }
            return bc.cards.exhaustPile[a.getSelectIdx()].id != sts::CardId::EXHUME;

        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
        case CardSelectTask::HEADBUTT:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.discardPile.size();

        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::DISCARD_ONE:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.cardsInHand;

        case CardSelectTask::ARMAMENTS:
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.cardsInHand) {
                return false;
            }
            return bc.cards.hand[a.getSelectIdx()].canUpgrade();

        case CardSelectTask::DUAL_WIELD: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.cardsInHand) {
                return false;
            }
            const auto &c = bc.cards.hand[a.getSelectIdx()];
            return c.getType() == sts::CardType::POWER || c.getType() == sts::CardType::ATTACK;
        }

        case CardSelectTask::SECRET_TECHNIQUE: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.drawPile.size()) {
                return false;
            }
            return bc.cards.drawPile[a.getSelectIdx()].getType() == CardType::SKILL;
        }

        case CardSelectTask::SECRET_WEAPON: {
            if (a.getSelectIdx() < 0 || a.getSelectIdx() >= bc.cards.drawPile.size()) {
                return false;
            }
            return bc.cards.drawPile[a.getSelectIdx()].getType() == CardType::ATTACK;
        }

        case CardSelectTask::SEEK:
        case CardSelectTask::OMNISCIENCE:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < bc.cards.drawPile.size();

        case CardSelectTask::WISH:
            return a.getSelectIdx() >= 0 && a.getSelectIdx() < 3;

        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::GAMBLE:
        case CardSelectTask::MEDITATE:
        default:
            return false;
    }
}

bool isValidMultiCardSelectAction(const BattleContext &bc, const search::Action &a) { // this can be sped up using compiler intrinsics
    switch (bc.cardSelectInfo.cardSelectTask) {
        case sts::CardSelectTask::EXHAUST_MANY: {
            const auto selected = a.getSelectedIdxs();
            if (selected.size() > bc.cardSelectInfo.pickCount) {
                return false;
            }
            for (auto x : selected) {
                if (x >= bc.cards.cardsInHand) {
                    return false;
                }
            }
            return true;
        }

        case sts::CardSelectTask::GAMBLE: {
            const auto selected = a.getSelectedIdxs();
            for (auto x : selected) {
                if (x >= bc.cards.cardsInHand) {
                    return false;
                }
            }
            return true;
        }

        case sts::CardSelectTask::SCRY: {
            const auto selected = a.getSelectedIdxs();
            for (auto x : selected) {
                if (x >= bc.cardSelectInfo.pickCount) return false;
            }
            return true;
        }

        default:
            return false;
    }
}

bool search::Action::isValidAction(const BattleContext &bc) const {
    if (bc.outcome != Outcome::UNDECIDED) {
        return false;
    }

    switch (getActionType()) {
        case ActionType::CARD:
            return isValidCardAction(bc, *this);

        case ActionType::POTION:
            return isValidPotionAction(bc, *this);

        case ActionType::SINGLE_CARD_SELECT:
            return isValidSingleCardSelectAction(bc, *this);

        case ActionType::MULTI_CARD_SELECT:
            return isValidMultiCardSelectAction(bc, *this);

        case ActionType::END_TURN:
            return bc.inputState == InputState::PLAYER_NORMAL;

        case ActionType::CHOOSE_STANCE:
            return bc.inputState == InputState::CHOOSE_STANCE_ACTION &&
                   (getSelectIdx() == static_cast<int>(Stance::CALM) ||
                    getSelectIdx() == static_cast<int>(Stance::WRATH));

        case ActionType::CHOOSE_ENEMY:
            return bc.inputState == InputState::SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON &&
                   getSelectIdx() >= 0 && getSelectIdx() < bc.monsters.monsterCount &&
                   bc.monsters.arr[getSelectIdx()].isTargetable();

        default:
            return false;
    }
}

std::ostream& printSingleCardSelectDescHelper(std::ostream &os, const BattleContext &bc, const search::Action &a) {
    // action is known to be valid here
    os << "{ " << cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)];
    os << " (" << a.getSelectIdx() << ") ";
    // TODO we don't know if it's selecting from hand or discard
    // const auto &c = bc.cards.hand[a.getSelectIdx()];
    // c.printSimpleDesc(os);
    return os << " }";
}

std::ostream& printMultiCardSelectDescHelper(std::ostream &os, const BattleContext &bc, const search::Action &a) {
    // action is known to be valid here
    os << "{ " << cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)];

    const auto selected = a.getSelectedIdxs();
    if (selected.empty()) {
        return os << " none";
    }

    for (int i = 0; i < selected.size(); ++i) {
        os << " (" << a.getSelectIdx() << ") ";
        const auto &c = bc.cards.hand[selected[i]];
        c.printSimpleDesc(os);
        if (i+1 < selected.size()) {
            os << ',';
        }
    }

    return os << " }";
}

std::ostream& search::Action::printDesc(std::ostream &os, const BattleContext &bc) const {
    if (!isValidAction(bc)) {
        return os << "{ INVALID ACTION }";
    }

    switch (getActionType()) {
        case ActionType::CARD: {
            const auto &c = bc.cards.hand[getSourceIdx()];
            os << "{ use card (" << getSourceIdx() << ") " << c;
            if (c.requiresTarget()) {
                const auto &m = bc.monsters.arr[getTargetIdx()];
                os << " -> (" << getTargetIdx() << ") " << m.getName();
            }
            return os << " }";
        }

        case ActionType::POTION: {
            const auto &p = bc.potions[getSourceIdx()];
            os << "{ ";
            if (getTargetIdx() == -1) {
                os << "discard potion ";
            } else {
                os << "drink potion ";
            }

            os << "(" << getSourceIdx() << ") " << getPotionName(p);

            if (potionRequiresTarget(p)) {
                const auto &m = bc.monsters.arr[getTargetIdx()];
                os << " -> (" << getTargetIdx() << ") " << m.getName();
            }
            return os << " }";
        }

        case ActionType::SINGLE_CARD_SELECT:
            return printSingleCardSelectDescHelper(os, bc, *this);

        case ActionType::MULTI_CARD_SELECT:
            return printMultiCardSelectDescHelper(os, bc, *this);

        case ActionType::END_TURN:
            return os << "{ end turn }";

        case ActionType::CHOOSE_STANCE:
            return os << "{ choose stance " << stanceStrings[getSelectIdx()] << " }";

        default:
            return os;
    }
}

void executeSingleCardSelectActionHelper(BattleContext &bc, search::Action a) {
    // assumed to be valid action here

    const auto idx = a.getSelectIdx();

    switch (bc.cardSelectInfo.cardSelectTask) {

        case CardSelectTask::ARMAMENTS:
            bc.chooseArmamentsCard(idx);
            break;

        case CardSelectTask::CODEX:
            if (idx != 3) { // idx 3 means skip
                bc.chooseCodexCard(bc.cardSelectInfo.codexCards()[idx]);
            }
            break;

        case CardSelectTask::DISCOVERY:
            bc.chooseDiscoveryCard(bc.cardSelectInfo.discovery_Cards()[idx]);
            break;

        case CardSelectTask::DUAL_WIELD:
            bc.chooseDualWieldCard(idx);
            break;

        case CardSelectTask::EXHAUST_ONE:
            bc.chooseExhaustOneCard(idx);
            break;

        case CardSelectTask::DISCARD_ONE:
            bc.chooseDiscardCard(idx);
            break;

        case CardSelectTask::EXHUME:
            bc.chooseExhumeCard(idx);
            break;

        case CardSelectTask::FORETHOUGHT:
            bc.chooseForethoughtCard(idx);
            break;

        case CardSelectTask::HEADBUTT:
            bc.chooseHeadbuttCard(idx);
            break;

        case CardSelectTask::HOLOGRAM:
            bc.chooseDiscardToHandCard(idx, false);
            break;

        case CardSelectTask::LIQUID_MEMORIES_POTION:
            bc.chooseDiscardToHandCard(idx, true);
            break;

        case CardSelectTask::SECRET_TECHNIQUE:
            bc.chooseDrawToHandCards(&idx, 1);
            break;

        case CardSelectTask::SECRET_WEAPON:
            bc.chooseDrawToHandCards(&idx, 1);
            break;

        case CardSelectTask::RECYCLE:
            bc.chooseRecycleCard(idx);
            break;

        case CardSelectTask::SEEK:
            bc.chooseSeekCard(idx);
            break;

        case CardSelectTask::OMNISCIENCE:
            bc.chooseOmniscienceCard(idx);
            break;

        case CardSelectTask::WISH:
            bc.chooseWish(idx);
            break;

        case CardSelectTask::SETUP:
            bc.chooseSetupCard(idx);
            break;

        case CardSelectTask::WARCRY:
            bc.chooseWarcryCard(idx);
            break;

        case CardSelectTask::NIGHTMARE:
            bc.chooseNightmareCard(idx);
            break;

        case CardSelectTask::MEDITATE:
        case CardSelectTask::INVALID:
        default:
            break;
    }
}

void executeMultiCardSelectActionHelper(BattleContext &bc, search::Action a) {
    // assumed to be valid action here
    switch (bc.cardSelectInfo.cardSelectTask) {

        case sts::CardSelectTask::EXHAUST_MANY:
            bc.chooseExhaustCards(a.getSelectedIdxs());
            break;

        case sts::CardSelectTask::GAMBLE:
            bc.chooseGambleCards(a.getSelectedIdxs());
            break;

        case sts::CardSelectTask::SCRY:
            bc.chooseScryCards(a.getSelectedIdxs());
            break;

        default:
            break;
    }
}

void search::Action::execute(BattleContext &bc) const {
#ifdef sts_asserts
    if (!isValidAction(bc)) {
        std::cerr << bc.seed << " " << static_cast<int>(getActionType()) << " " << getSourceIdx() << " " << getTargetIdx() << std::endl;
        std::cerr <<  cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)] << std::endl;
        std::cerr << bc << std::endl;
        assert(false);
    }
#endif

    switch (getActionType()) {
        case ActionType::CARD: {
            const CardQueueItem item(bc.cards.hand[getSourceIdx()], getTargetIdx(), bc.player.energy);
            bc.addToBotCard(item);
            break;
        }

        case ActionType::POTION: {
            if (getTargetIdx() > 5) {
                bc.discardPotion(getSourceIdx());
            } else {
                bc.drinkPotion(getSourceIdx(), getTargetIdx());
            }
            break;
        }

        case ActionType::SINGLE_CARD_SELECT:
            executeSingleCardSelectActionHelper(bc, *this);
            break;

        case ActionType::MULTI_CARD_SELECT:
            executeMultiCardSelectActionHelper(bc, *this);
            break;

        case ActionType::END_TURN:
            bc.endTurn();
            break;

        case ActionType::CHOOSE_STANCE:
            bc.addToBot(Actions::ChangeStance(static_cast<Stance>(getSelectIdx())));
            break;

        case ActionType::CHOOSE_ENEMY:
            bc.monsters.arr[getSelectIdx()].buff<MS::POISON>(bc.cardSelectInfo.data0);
            break;

        default:
            break;
    }

    bc.inputState = InputState::EXECUTING_ACTIONS;
    bc.executeActions();
}

template <typename ForwardIt>
void setupCardOptionsHelper(std::vector<search::Action> &actions, const ForwardIt begin, const ForwardIt end, const std::function<bool(const CardInstance &)> &p= nullptr) {
    for (int i = 0; begin+i != end; ++i) {
        const auto &c = begin[i];
        if (!p || (p(c))) {
            actions.emplace_back(search::ActionType::SINGLE_CARD_SELECT, i);
        }
    }
}

std::vector<search::Action> search::Action::enumerateCardSelectActions(const BattleContext &bc) {
    std::vector<search::Action> actions;

    switch (bc.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::ARMAMENTS:
            setupCardOptionsHelper( actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) { return c.canUpgrade(); });
            break;

        case CardSelectTask::CODEX:
            for (int i = 0; i < 4; ++i) { // i -> 3 action means skip
                actions.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DISCOVERY:
            for (int i = 0; i < 3; ++i) {
                actions.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::DUAL_WIELD:
            setupCardOptionsHelper( actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand,
                                    [] (const CardInstance &c) {
                                        return c.getType() == CardType::POWER || c.getType() == CardType::ATTACK;
                                    });
            break;

        case CardSelectTask::EXHUME:
            setupCardOptionsHelper(actions, bc.cards.exhaustPile.begin(), bc.cards.exhaustPile.end(),
                                   [](const auto &c) { return c.getId() != CardId::EXHUME; });
            break;

        case CardSelectTask::EXHAUST_ONE:
            setupCardOptionsHelper(actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::DISCARD_ONE:
            setupCardOptionsHelper(actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            setupCardOptionsHelper(actions, bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
            break;

        case CardSelectTask::HEADBUTT:
        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
            setupCardOptionsHelper(actions, bc.cards.discardPile.begin(), bc.cards.discardPile.end());
            break;

        case CardSelectTask::SECRET_TECHNIQUE:
            setupCardOptionsHelper(actions, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                   [] (const CardInstance &c) {
                                       return c.getType() == CardType::SKILL;
                                   });
            break;

        case CardSelectTask::SECRET_WEAPON:
            setupCardOptionsHelper(actions, bc.cards.drawPile.begin(), bc.cards.drawPile.end(),
                                   [] (const CardInstance &c) {
                                       return c.getType() == CardType::ATTACK;
                                   });
            break;

        case CardSelectTask::SEEK:
        case CardSelectTask::OMNISCIENCE:
            setupCardOptionsHelper(actions, bc.cards.drawPile.begin(), bc.cards.drawPile.end());
            break;

        case CardSelectTask::WISH:
            for (int i = 0; i < 3; ++i) {
                actions.push_back({Action(search::ActionType::SINGLE_CARD_SELECT, i)});
            }
            break;

        case CardSelectTask::EXHAUST_MANY:
        case CardSelectTask::GAMBLE:
        case CardSelectTask::SCRY: {
            const auto selectionCount = bc.cardSelectInfo.cardSelectTask == CardSelectTask::SCRY
                ? bc.cardSelectInfo.pickCount : bc.cards.cardsInHand;
            const auto subsetCount = 1 << selectionCount;
            actions.reserve(subsetCount);
            for (int selectedBits = 0; selectedBits < subsetCount; ++selectedBits) {
                Action action(ActionType::MULTI_CARD_SELECT, selectedBits);
                if (action.isValidAction(bc)) {
                    actions.push_back(action);
                }
            }
            break;
        }

        default:
#ifdef sts_asserts
            assert(false);
#endif
            break;
    }
    return actions;
}

std::vector<search::Action> search::Action::getAllActionsInState(const BattleContext &bc) {
    std::vector<Action> actions;
    if (bc.outcome != Outcome::UNDECIDED) {
        return actions;
    }

    if (bc.inputState == InputState::CARD_SELECT) {
        return enumerateCardSelectActions(bc);
    }

    if (bc.inputState == InputState::CHOOSE_STANCE_ACTION) {
        actions.emplace_back(ActionType::CHOOSE_STANCE, static_cast<int>(Stance::CALM));
        actions.emplace_back(ActionType::CHOOSE_STANCE, static_cast<int>(Stance::WRATH));
        return actions;
    }

    if (bc.inputState == InputState::SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON) {
        for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
            Action action(ActionType::CHOOSE_ENEMY, targetIdx);
            if (action.isValidAction(bc)) actions.push_back(action);
        }
        return actions;
    }

    if (bc.inputState != InputState::PLAYER_NORMAL) {
        return actions;
    }

    for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
        const auto &card = bc.cards.hand[handIdx];
        if (card.requiresTarget()) {
            for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                Action action(ActionType::CARD, handIdx, targetIdx);
                if (action.isValidAction(bc)) {
                    actions.push_back(action);
                }
            }
        } else {
            Action action(ActionType::CARD, handIdx, 0);
            if (action.isValidAction(bc)) {
                actions.push_back(action);
            }
        }
    }

    for (int potionIdx = 0; potionIdx < bc.potionCapacity; ++potionIdx) {
        const auto potion = bc.potions[potionIdx];
        if (potion == Potion::EMPTY_POTION_SLOT || potion == Potion::INVALID) {
            continue;
        }

        // A target index above 5 encodes discard in the legacy action format.
        Action discard(ActionType::POTION, potionIdx, 6);
        if (discard.isValidAction(bc)) {
            actions.push_back(discard);
        }

        if (potionRequiresTarget(potion)) {
            for (int targetIdx = 0; targetIdx < bc.monsters.monsterCount; ++targetIdx) {
                Action action(ActionType::POTION, potionIdx, targetIdx);
                if (action.isValidAction(bc)) {
                    actions.push_back(action);
                }
            }
        } else {
            Action action(ActionType::POTION, potionIdx, 0);
            if (action.isValidAction(bc)) {
                actions.push_back(action);
            }
        }
    }

    Action endTurn(ActionType::END_TURN);
    if (endTurn.isValidAction(bc)) {
        actions.push_back(endTurn);
    }
    return actions;
}








} // namespace sts
