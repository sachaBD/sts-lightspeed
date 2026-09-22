//
// Created by keega on 9/18/2021.
//

#include "sim/search/BattleScumSearcher2.h"
#include "sim/search/ExpertKnowledge.h"
#include "sim/search/SimpleAgent.h"

#include <utility>
#include <string>
#include <memory>

namespace sts::search {

thread_local std::int64_t simulationIdx = 0; // for debugging

thread_local search::BattleScumSearcher2 *g_debug_scum_search;



search::BattleScumSearcher2::BattleScumSearcher2(const BattleContext &bc, search::EvalFnc _evalFnc)
    : rootState(new BattleContext(bc)), evalFnc(std::move(_evalFnc)), randGen(bc.seed+bc.floorNum) {
}

void search::BattleScumSearcher2::search(int64_t simulations) {
    g_debug_scum_search = this;

    if (isTerminalState(*rootState)) {
        auto evaluation = evaluateEndState(*rootState);
        outcomePlayerHp = rootState->player.curHp;
        bestActionSequence = {};

        root.evaluationSum = evaluation;
        root.simulationCount = 1;
    }

    for (std::int64_t simCount = 0; simCount < simulations; ++simCount) {
        step();
    }
}

void search::BattleScumSearcher2::step() {
    searchStack = {&root};
    actionStack.clear();
    BattleContext curState;
    curState = *rootState;

    while (true) {
        auto &curNode = *searchStack.back();

        if (isTerminalState(curState)) {
            updateFromPlayout(searchStack, actionStack, curState);
            return;
        }

        const bool isLeaf = curNode.edges.empty();
        if (isLeaf) {

            ++simulationIdx;
            enumerateActionsForNode(curNode, curState);
            const auto selectIdx = selectFirstActionForLeafNode(curNode);
            auto &edgeTaken = curNode.edges[selectIdx];

//            edgeTaken.action.printDesc(std::cout, curState) << std::endl;
            edgeTaken.action.execute(curState);

            actionStack.push_back(edgeTaken.action);
            searchStack.push_back(&edgeTaken.node);

            playoutRandom(curState, actionStack);
            updateFromPlayout(searchStack, actionStack, curState);
            return;

        } else {
            const auto selectIdx = selectBestEdgeToSearch(curNode);
            auto &edgeTaken = curNode.edges[selectIdx];

//            edgeTaken.action.printDesc(std::cout, curState) << std::endl;
            edgeTaken.action.execute(curState);

            actionStack.push_back(edgeTaken.action);
            searchStack.push_back(&edgeTaken.node);
        }
    }
}

void search::BattleScumSearcher2::updateFromPlayout(const std::vector<Node *> &stack, const std::vector<Action> &actionStack, const BattleContext &endState) {
    const auto evaluation = evaluateEndState(endState);

    if (evaluation > bestActionValue) {
        bestActionSequence = actionStack;
        bestActionValue = evaluation;
        outcomePlayerHp = endState.player.curHp;
    }

    if (evaluation < minActionValue) {
        minActionValue = evaluation;
    }

    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        auto &node = *(*it);
        ++node.simulationCount;
        node.evaluationSum += evaluation;
        node.bestEvaluation = std::max(node.bestEvaluation, evaluation);
        if (endState.outcome == Outcome::PLAYER_VICTORY) {
            node.bestWinningHp = std::max(node.bestWinningHp, endState.player.curHp);
        }
    }
}

bool search::BattleScumSearcher2::isTerminalState(const BattleContext &bc) const { // maybe can optimize by making this evaluate directly if score cannot possibly be higher than best
    return bc.outcome != Outcome::UNDECIDED;
}

double search::BattleScumSearcher2::evaluateEdge(const search::BattleScumSearcher2::Node &parent, int edgeIdx) {

    const auto &edge = parent.edges[edgeIdx];

    double qualityValue = 0;
    if (!bestActionSequence.empty()) {
        auto avgEvaluation = edge.node.evaluationSum / (edge.node.simulationCount+1);
        if (backupMode == 1 && edge.node.simulationCount > 0) {
            avgEvaluation = edge.node.bestEvaluation;
        }
        double evalRange = std::max(1e-9, bestActionValue - minActionValue);
        qualityValue = avgEvaluation / evalRange;
    }

    double explorationValue = explorationParameter *
            std::sqrt(std::log(parent.simulationCount+1) / (edge.node.simulationCount+1));

    return qualityValue + explorationValue;
}

int search::BattleScumSearcher2::selectBestEdgeToSearch(const search::BattleScumSearcher2::Node &cur) {
    if (cur.edges.size() == 1) {
        return 0;
    }

    auto bestEdge = 0;
    auto bestEdgeValue = evaluateEdge(cur, bestEdge);

    for (int i = 1; i < cur.edges.size(); ++i) {
        const auto value = evaluateEdge(cur, i);
        if (value > bestEdgeValue) {
            bestEdge = i;
            bestEdgeValue = value;
        }
    }
    return bestEdge;
}

int search::BattleScumSearcher2::selectFirstActionForLeafNode(const search::BattleScumSearcher2::Node &leafNode) {
    auto dist = std::uniform_int_distribution<int>(0, static_cast<int>(leafNode.edges.size())-1);
    return dist(randGen);
}

void search::BattleScumSearcher2::playoutRandom(BattleContext &state, std::vector<Action> &actionStack) {
    Node tempNode; // temp
    while (!isTerminalState(state)) {
        ++simulationIdx;
        enumerateActionsForNode(tempNode, state);
        if (tempNode.edges.empty()) {
            std::cerr << state.seed << " " << simulationIdx << std::endl;
            std::cerr << state.monsters.arr[0].getName() << " " << state.floorNum << " " << monsterEncounterStrings[static_cast<int>(state.encounter)] << std::endl;
            assert(false);
        }

        const int selectedIdx = selectRolloutAction(tempNode, state);

        const auto action = tempNode.edges[selectedIdx].action;
//        action.printDesc(std::cout, state) << std::endl;
        actionStack.push_back(action);
        action.execute(state);

        tempNode.edges.clear();
    }
}

void search::BattleScumSearcher2::enumerateActionsForNode(search::BattleScumSearcher2::Node &node,
                                                               const BattleContext &bc) {
    switch (bc.inputState) {
        case InputState::PLAYER_NORMAL:
            enumerateCardActions(node, bc);
            enumeratePotionActions(node, bc);
            node.edges.push_back({Action(ActionType::END_TURN)});
            break;

        case InputState::CARD_SELECT:
            enumerateCardSelectActions(node, bc);
            break;

        default:
#ifdef sts_asserts
            std::cerr << "enumerateActionsForNode: invalid input state: " << static_cast<int>(bc.inputState) << std::endl;
            assert(false);
#endif
            break;
    }

#ifdef sts_print_debug
    std::cout << "{ (" << node.edges.size() << ") ";
    for (int i = 0; i < node.edges.size(); ++i) {
        node.edges[i].action.printDesc(std::cout, bc) << ", ";
    }
    std::cout << " }" << std::endl;
#endif
}

void search::BattleScumSearcher2::enumerateCardActions(search::BattleScumSearcher2::Node &node,
                                                            const BattleContext &bc) {
    if (!bc.isCardPlayAllowed()) {
        return;
    }

    fixed_list<std::pair<int,int>, 10> playableHandIdxs;
    for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
        const auto &c = bc.cards.hand[handIdx];
        if (!c.canUseOnAnyTarget(bc)) {
            continue;
        }

        bool isUniqueAction = true;

        if (handIdx > 0) {
            const auto &lastCard = bc.cards.hand[handIdx-1];

            bool isEqualToLastCard = c.id == lastCard.id &&
                    c.getUpgradeCount() == lastCard.getUpgradeCount() &&
                    // both should be less than deck size c.uniqueId < bc.cards.deck
                    c.costForTurn == lastCard.costForTurn &&
                    c.cost == lastCard.cost &&
                    c.freeToPlayOnce == lastCard.freeToPlayOnce &&
                    c.specialData == lastCard.specialData;

            if (isEqualToLastCard) {
                isUniqueAction = false;
            }
        }

        if (isUniqueAction) {
            playableHandIdxs.push_back( {handIdx, search::Expert::getPlayOrdering(c.getId())} );
        }
    }

    std::sort(playableHandIdxs.begin(), playableHandIdxs.end(), [](auto a, auto b) { return a.second < b.second; });

    for (auto pair : playableHandIdxs) {
        const auto handIdx = pair.first;
        const auto &c = bc.cards.hand[handIdx];

        if (c.requiresTarget()) {
            for (int tIdx = bc.monsters.monsterCount-1; tIdx >= 0; --tIdx) {
                if (!bc.monsters.arr[tIdx].isTargetable()) {
                    continue;
                }
                node.edges.push_back({Action(ActionType::CARD, handIdx, tIdx)});
            }
        } else {
            node.edges.push_back({Action(ActionType::CARD, handIdx)});
        }
    }

}

void search::BattleScumSearcher2::enumeratePotionActions(search::BattleScumSearcher2::Node &node,
                                                              const BattleContext &bc) {

    const auto hasValidTarget = bc.monsters.getTargetableCount() > 0;

    for (int pIdx = 0; pIdx < bc.potionCapacity; ++pIdx) {

        const auto p = bc.potions[pIdx];
        if (p == Potion::EMPTY_POTION_SLOT || p == Potion::INVALID) {
            continue;
        }

        if (p == Potion::FAIRY_POTION) {
            const Action discard(ActionType::POTION, pIdx, 6);
            if (discard.isValidAction(bc)) {
                node.edges.push_back({discard});
            }
            continue;
        }

        bool addedUse = false;
        if (!potionRequiresTarget(p)) {
            const Action use(ActionType::POTION, pIdx, 0);
            if (use.isValidAction(bc)) {
                node.edges.push_back({use});
                addedUse = true;
            }
        } else if (hasValidTarget) {
            for (int tIdx = 0; tIdx < bc.monsters.monsterCount; ++tIdx) {
                const Action use(ActionType::POTION, pIdx, tIdx);
                if (use.isValidAction(bc)) {
                    node.edges.push_back({use});
                    addedUse = true;
                }
            }
        }

        // Keep a legal way to clear unusable potions (Fairy, a target potion
        // with no target, and Smoke Bomb in a boss encounter). Never seed the
        // search tree with an action the public environment would reject.
        if (!addedUse) {
            const Action discard(ActionType::POTION, pIdx, 6);
            if (discard.isValidAction(bc)) {
                node.edges.push_back({discard});
            }
        }
    }
}

namespace {
double keepForEngine(const CardInstance &card, const BattleContext &state) {
    if (card.getType() == CardType::STATUS || card.getType() == CardType::CURSE) return -20;
    switch (card.id) {
        case CardId::STRIKE_RED: return 0;
        case CardId::DEFEND_RED: return state.player.hasStatus<PS::CORRUPTION>() ? 3 : 1;
        case CardId::DARK_EMBRACE: return state.player.hasStatus<PS::DARK_EMBRACE>() ? 6 : 22;
        case CardId::FEEL_NO_PAIN: return 19;
        case CardId::CORRUPTION: return state.player.hasStatus<PS::CORRUPTION>() ? -8 : 20;
        case CardId::BARRICADE: return state.player.hasStatus<PS::BARRICADE>() ? -8 : 16;
        case CardId::ENTRENCH: return state.player.hasStatus<PS::BARRICADE>() ? 22 : 4;
        case CardId::BODY_SLAM: return state.player.hasStatus<PS::BARRICADE>() ? 22 : 8;
        case CardId::SECOND_WIND: case CardId::FIEND_FIRE: return 18;
        case CardId::POWER_THROUGH: case CardId::BURNING_PACT: return 13;
        case CardId::REAPER: case CardId::OFFERING: return 20;
        case CardId::POMMEL_STRIKE: case CardId::SHRUG_IT_OFF: return 12;
        case CardId::DROPKICK: case CardId::SPOT_WEAKNESS: case CardId::LIMIT_BREAK: return 14;
        default: return 5 + 2 * card.isUpgraded();
    }
}
}

int search::BattleScumSearcher2::engineRolloutAction(const Node &node, const BattleContext &state) {
    if (state.player.hasRelic<RelicId::RUNIC_DOME>()) return -1;
    if (state.inputState == InputState::CARD_SELECT) {
        const auto task = state.cardSelectInfo.cardSelectTask;
        const bool exhaust = task == CardSelectTask::EXHAUST_ONE || task == CardSelectTask::EXHAUST_MANY;
        const bool discard = task == CardSelectTask::GAMBLE || task == CardSelectTask::DISCARD_ONE;
        const bool retrieve = task == CardSelectTask::HEADBUTT || task == CardSelectTask::LIQUID_MEMORIES_POTION;
        const bool exhume = task == CardSelectTask::EXHUME;
        const bool upgrade = task == CardSelectTask::ARMAMENTS;
        if (!exhaust && !discard && !retrieve && !exhume && !upgrade) return -1;
        int selected = -1;
        double best = -1e20;
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            const auto action = node.edges[i].action;
            double value = 0;
            if (action.getActionType() == ActionType::MULTI_CARD_SELECT) {
                for (const auto index : action.getSelectedIdxs())
                    value += 2 - keepForEngine(state.cards.hand[index], state);
            } else if (action.getActionType() == ActionType::SINGLE_CARD_SELECT) {
                const auto index = action.getSelectIdx();
                const auto &card = retrieve ? state.cards.discardPile[index]
                    : exhume ? state.cards.exhaustPile[index] : state.cards.hand[index];
                value = keepForEngine(card, state) * (exhaust || discard ? -1 : 1);
                if (upgrade && (card.id == CardId::TRUE_GRIT || card.id == CardId::POMMEL_STRIKE)) value += 12;
            } else continue;
            if (value > best) { best = value; selected = static_cast<int>(i); }
        }
        return selected;
    }
    if (state.inputState != InputState::PLAYER_NORMAL) return -1;
    int skills = 0, naturalExhaust = 0, incoming = 0;
    // Membership/counts only: hidden draw order and future RNG never enter the guide.
    const auto count = [&](const CardInstance &card) {
        skills += card.getType() == CardType::SKILL;
        naturalExhaust += card.doesExhaust();
    };
    for (int i = 0; i < state.cards.cardsInHand; ++i) count(state.cards.hand[i]);
    for (const auto &card : state.cards.drawPile) count(card);
    for (const auto &card : state.cards.discardPile) count(card);
    for (int i = 0; i < state.monsters.monsterCount; ++i) {
        const auto &monster = state.monsters.arr[i];
        if (!monster.isTargetable()) continue;
        const auto damage = monster.getMoveBaseDamage(state);
        incoming += monster.calculateDamageToPlayer(state, damage.damage) * damage.attackCount;
    }
    const bool safe = state.player.curHp + state.player.block > incoming + 8;
    int selected = -1;
    double best = 0;
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        const auto action = node.edges[i].action;
        if (action.getActionType() != ActionType::CARD) continue;
        const auto &card = state.cards.hand[action.getSourceIdx()];
        const auto cost = card.isFreeToPlay(state) ? 0 : std::max(0, static_cast<int>(card.costForTurn));
        const int remaining = state.player.energy - cost;
        int immediateExhaust = 0;
        bool wind = false, through = false;
        for (int h = 0; h < state.cards.cardsInHand; ++h) {
            if (h == action.getSourceIdx()) continue;
            const auto &other = state.cards.hand[h];
            const auto otherCost = other.isFreeToPlay(state) ? 0 : std::max(0, static_cast<int>(other.costForTurn));
            if (otherCost > remaining || !other.canUseOnAnyTarget(state)) continue;
            const bool skill = other.getType() == CardType::SKILL;
            immediateExhaust += other.doesExhaust() || (skill && state.player.hasStatus<PS::CORRUPTION>());
            if (other.id == CardId::SECOND_WIND || other.id == CardId::FIEND_FIRE) immediateExhaust += 2;
            wind |= other.id == CardId::SECOND_WIND;
            through |= other.id == CardId::POWER_THROUGH;
        }
        double value = 0;
        switch (card.id) {
            case CardId::FEEL_NO_PAIN:
                if (immediateExhaust || (safe && naturalExhaust >= 3)) value = 120;
                break;
            case CardId::DARK_EMBRACE:
                if (immediateExhaust || (safe && (naturalExhaust >= 3 || state.player.hasStatus<PS::CORRUPTION>()))) value = 115;
                break;
            case CardId::CORRUPTION:
                if (skills >= 4 && (safe || immediateExhaust)) value = 105;
                break;
            case CardId::BARRICADE:
                if (safe && (state.player.block >= 15 || state.player.hasStatus<PS::FEEL_NO_PAIN>())) value = 100;
                break;
            case CardId::POWER_THROUGH:
                if (wind) value = 80;
                break;
            case CardId::ENTRENCH:
                if (state.player.hasStatus<PS::BARRICADE>() && state.player.block >= 15) value = 75 + state.player.block * .02;
                break;
            case CardId::SECOND_WIND:
                if (!through && (state.player.hasStatus<PS::DARK_EMBRACE>() || state.player.hasStatus<PS::FEEL_NO_PAIN>())) value = 60;
                break;
            default: break;
        }
        if (value > best) { best = value; selected = static_cast<int>(i); }
    }
    return selected;
}

int search::BattleScumSearcher2::selectRolloutAction(const Node &node, const BattleContext &state) {
    if (rolloutMode == 3 && std::bernoulli_distribution(0.8)(randGen)) {
        const int preferred = engineRolloutAction(node, state);
        if (preferred >= 0) return preferred;
    }
    if (rolloutMode == 0 || state.inputState != InputState::PLAYER_NORMAL) {
        return std::uniform_int_distribution<int>(0, static_cast<int>(node.edges.size()) - 1)(randGen);
    }
    if ((rolloutMode == 2 || rolloutMode == 3)
        && !state.player.hasRelic<RelicId::RUNIC_DOME>()
        && std::bernoulli_distribution(0.8)(randGen)) {
        // The heuristic sees only the current hand and public intent. It does
        // not inspect shuffle/RNG state. Stochastic support remains available
        // to discover actions/combinations omitted by that weak policy.
        thread_local SimpleAgent guide;
        thread_local GameContext context;
        context.act = std::clamp((state.floorNum - 1) / 17 + 1, 1, 3);
        guide.curGameContext = &context;
        const auto bits = guide.suggestBattleCardPlay(state).bits;
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            if (node.edges[i].action.bits == bits) return static_cast<int>(i);
        }
    }
    // A rollout policy, not a legal-action filter. Ending the turn retains
    // positive support (important for Spikers, curses and self-damage cards),
    // but does not get the same mass as using an available card. This tests
    // the large number of deliberately wasted turns in uniform playouts.
    std::vector<double> weights;
    weights.reserve(node.edges.size());
    for (const auto &edge : node.edges) {
        const auto type = edge.action.getActionType();
        weights.push_back(type == ActionType::END_TURN ? 0.03
            : type == ActionType::POTION ? 0.20 : 1.0);
    }
    return std::discrete_distribution<int>(weights.begin(), weights.end())(randGen);
}

template <typename ForwardIt>
void setupCardOptionsHelper(search::BattleScumSearcher2::Node &node, const ForwardIt begin, const ForwardIt end, const std::function<bool(const CardInstance &)> &p= nullptr) {
    for (int i = 0; begin+i != end; ++i) {
        const auto &c = begin[i];
        if (!p || (p(c))) {
            node.edges.push_back(
                    {search::Action(search::ActionType::SINGLE_CARD_SELECT, i)}
                );
        }
    }
}

void search::BattleScumSearcher2::enumerateCardSelectActions(
        search::BattleScumSearcher2::Node &node, const BattleContext &bc) {
    // Use the same complete legal support as execution. The former separate
    // enumerator silently reduced every multi-card discard/exhaust choice to
    // the empty subset and omitted several supported selection tasks.
    for (const auto &action : Action::enumerateCardSelectActions(bc)) {
        if (action.isValidAction(bc)) node.edges.push_back({action});
    }
}

double getNonMinionMonsterCurHpRatio(const BattleContext &bc) {
    int curHpTotal = 0;
    int maxHpTotal = 0;

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &m = bc.monsters.arr[i];
        if (!m.hasStatus<MS::MINION>() && m.id != sts::MonsterId::INVALID) {
            curHpTotal += m.curHp;
            maxHpTotal += m.maxHp;
        }
    }

    if (curHpTotal == 0 || maxHpTotal == 0) {
        return 0;
    }

    return (double)curHpTotal / maxHpTotal;
}

double search::BattleScumSearcher2::evaluateEndState(const BattleContext &bc) {
    double potionScore = bc.potionCount * 4;

    if (bc.outcome == Outcome::PLAYER_VICTORY) {
        return 100 * (35 + bc.player.curHp + potionScore - (bc.turn * 0.01));

    } else {
//        double statusScore =
//                (bc.player.getStatus<PS::STRENGTH>() * .5);
        const bool couldHaveSpikers = bc.encounter == MonsterEncounter::THREE_SHAPES || bc.encounter == MonsterEncounter::FOUR_SHAPES;
        double energyPenalty = bc.energyWasted * -0.2 * (couldHaveSpikers ? 0 : 1);
        double drawBonus = bc.cardsDrawn * 0.03;
        double aliveScore = bc.monsters.monstersAlive*-1;

        return (1-getNonMinionMonsterCurHpRatio(bc))*10 + aliveScore + energyPenalty + drawBonus + potionScore / 2 + (bc.turn * .2);
    }
}

struct LayerStruct {
    const search::BattleScumSearcher2::Node *node;
    BattleContext *bc;
    int edgeIdx;
};

typedef std::pair<search::BattleScumSearcher2::Edge, std::unique_ptr<const BattleContext>> EdgeInfo;

std::vector<EdgeInfo> getEdgesForLayer(const search::BattleScumSearcher2 &s, int layerNum) {
    if (layerNum <= 0) {
        return {};
    }

    std::vector<EdgeInfo> layerEdges;

    std::vector<LayerStruct> curStack { {&s.root, new BattleContext(*s.rootState), 0} };

    while (!curStack.empty()) {
        if (curStack.size() == layerNum) {
            for (const auto &edge : curStack.back().node->edges) {
                layerEdges.emplace_back(edge, new BattleContext(*curStack.back().bc));
            }
        }

       // curStack size less than layerNum
       const bool visitedAll = curStack.back().edgeIdx >= curStack.back().node->edges.size();
       if (visitedAll || curStack.size() == layerNum) {
           delete curStack.back().bc;
           curStack.pop_back();
           continue;
       }

        // visit next edge
        auto &nextIdx = curStack.back().edgeIdx;
        const auto action = curStack.back().node->edges[nextIdx].action;

        BattleContext bc(*curStack.back().bc);
        action.execute(bc);

        curStack.push_back( {&curStack.back().node->edges[nextIdx++].node, new BattleContext(bc), 0} );
    }

    return layerEdges;
}

void search::BattleScumSearcher2::printSearchTree(std::ostream &os, int levels) {
    std::vector<std::vector<EdgeInfo>> layerEdges;
    for (int depth = 1; depth <= levels; ++depth) {
        layerEdges.push_back(getEdgesForLayer(*this, depth));
    }

//    auto maxIt = std::max(layerEdges.begin(), layerEdges.end(), [](auto a, auto b) { return a->size() < b->size(); });
//    if (maxIt == layerEdges.end()) {
//        return;
//    }
//    // maxIt points to something
//    const auto maxSize = maxIt->size();
//    constexpr auto edgeWidth = 30;

    for (int depth = 0; depth < levels; ++depth) {
        for (const auto &x : layerEdges[depth]) {
            os << "(" << x.first.node.simulationCount << ")";
            x.first.action.printDesc(os, *x.second) << "\t";
        }
        std::cout << '\n';
    }

}

void search::BattleScumSearcher2::printSearchStack(std::ostream &os, bool skipLast) {
    for (int i = 0; i < actionStack.size(); ++i) {
        const auto &a = actionStack[i];
        os << std::hex << a.bits << '\n';
    }

    os.flush();

//    BattleContext curBc = *rootState;
//    os << "explorationParameter: " << explorationParameter << '\n';
//    os << "bestActionValue: " << bestActionValue << '\n';
//    os << "minActionValue: " << minActionValue << '\n';
//    os << "outcomePlayerHp: " << outcomePlayerHp << '\n';
//    os << "root node:\n";
//    os << curBc << "\n";
//
//    for (int i = 0; i < actionStack.size(); ++i) {
//        if (i < searchStack.size()) {
//            const auto &n = searchStack[i];
//            os << i << " nodeSearched: " << n->simulationCount << " { ";
//            for (const auto &edge : n->edges) {
//                os << "(" << edge.node.simulationCount << ")";
//                edge.action.printDesc(os, curBc) << " ";
//            }
//            os << "}\n";
//        }
//
//        const auto &a = actionStack[i];
//        os << i << " actionTaken: ";
//        a.printDesc(os, curBc) << '\n';
//
//        if (skipLast && (i + 1 >= actionStack.size())) {
//            break;
//        }
//
//        a.execute(curBc);
//        os << curBc << '\n';
//    }
//
//    os.flush();
}

} // namespace sts::search
