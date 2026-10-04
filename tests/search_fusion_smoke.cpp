#include "game/GameContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace sts;
using namespace sts::search;
namespace {
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
BattleContext jawWorm(std::uint64_t seed) {
    GameContext game(CharacterClass::IRONCLAD, seed, 0);
    game.floorNum = 1;
    BattleContext state; state.init(game, MonsterEncounter::JAW_WORM);
    check(state.turn == 0 && isSettledSearchState(state), "fixture must start at turn one decision one");
    return state;
}
search::Action playAction(const BattleContext &s, CardId card) {
    for (const auto action : search::Action::getAllActionsInState(s))
        if (action.getActionType() == ActionType::CARD && s.cards.hand[action.getSourceIdx()].id == card)
            return action;
    throw std::runtime_error("fixture card not playable");
}
BattleContext play(BattleContext s, CardId card) { playAction(s, card).execute(s); return s; }
struct Counts {
    std::size_t prefixes = 0;
    std::set<std::vector<std::uint64_t>> unique;
};
void enumerate(BattleContext state, bool slots, Counts &counts) {
    ++counts.prefixes;
    counts.unique.insert(publicSearchWords(state));
    std::set<std::uint64_t> actions;
    for (const auto action : search::Action::getAllActionsInState(state)) {
        if (action.getActionType() != ActionType::CARD) continue;
        if (!slots && !actions.insert(semantic_action_key(state, action)).second) continue;
        auto child = state; action.execute(child);
        check(child.turn == 0 && isSettledSearchState(child), "unexpected turn/selection in starter-deck enumeration");
        enumerate(std::move(child), slots, counts);
    }
}
const PublicBeliefCombatSearch::Edge &cardEdge(const PublicBeliefCombatSearch::Node &node,
                                             const BattleContext &state, CardId id) {
    const auto semantic = semantic_action_key(state, playAction(state, id));
    for (const auto &edge : node.edges) if (edge.semanticKey == semantic) return edge;
    throw std::runtime_error("card edge missing");
}
}
int main() {
    const auto root = jawWorm(0);
    std::cout << "Jaw Worm, default Ironclad deck, A0, seed=0, turn=1 (native zero-based): ";
    for (int i = 0; i < root.cards.cardsInHand; ++i) std::cout << root.cards.hand[i].getName() << ',';
    std::cout << '\n';
    const auto afterStrike = play(root, CardId::STRIKE_RED);
    const auto afterDefend = play(root, CardId::DEFEND_RED);
    const auto sd = play(afterStrike, CardId::DEFEND_RED);
    const auto ds = play(afterDefend, CardId::STRIKE_RED);
    check(publicSearchWords(sd) == publicSearchWords(ds), "Strike/Defend must share a decision problem");
    check(publicSearchWords(play(play(root, CardId::BASH), CardId::STRIKE_RED))
        != publicSearchWords(play(play(root, CardId::STRIKE_RED), CardId::BASH)), "Bash/Strike must not fuse");
    auto permuted = root; std::swap(permuted.cards.hand[0], permuted.cards.hand[1]);
    check(public_search_key(permuted) == public_search_key(root), "hand order must not distinguish nodes");
    auto changed = root; changed.player.stance = Stance::WRATH;
    check(public_search_key(changed) != public_search_key(root), "stance must distinguish nodes");
    changed = root; ++changed.player.echoFormCardsDoubled;
    check(public_search_key(changed) != public_search_key(root), "trigger counters must distinguish nodes");
    changed = root; changed.cardRandomRng.random();
    check(public_search_key(changed) == public_search_key(root), "hidden RNG seeds must not block evidence reuse");
    int other = 1;
    while (other < static_cast<int>(root.cards.drawPile.size())
           && root.cards.drawPile[other].id == root.cards.drawPile[0].id) ++other;
    check(other < static_cast<int>(root.cards.drawPile.size()), "draw-order fixture needs distinct cards");
    changed = root; std::swap(changed.cards.drawPile[0], changed.cards.drawPile[other]);
    check(public_search_key(changed) == public_search_key(root), "unseen draw order must not block sharing");
    auto bossMemory = root;
    bossMemory.monsters.arr[0].id = MonsterId::TIME_EATER;
    changed = bossMemory; changed.monsters.arr[0].miscInfo = 1;
    check(public_search_key(changed) != public_search_key(bossMemory), "visible one-off Haste history must distinguish nodes");
    changed = root; ++changed.monsters.arr[0].miscInfo;
    check(public_search_key(changed) == public_search_key(root), "arbitrary private enemy scratch must not enter identity");
    auto eye = root; eye.player.setHasRelic<RelicId::FROZEN_EYE>(true);
    changed = eye; std::swap(changed.cards.drawPile[0], changed.cards.drawPile[other]);
    check(public_search_key(changed) != public_search_key(eye), "Frozen Eye order must distinguish nodes");

    auto known = root;
    known.cards.moveToDrawPileTop(CardInstance(CardId::BASH, true));
    known.cards.insertToDrawPile(0, CardInstance(CardId::DEFEND_RED, true));
    for (int i = 0; i < 5; ++i) {
        auto sampled = known;
        PublicBeliefCombatSearch::resampleDraw(sampled, known, 100 + i);
        check(sampled.cards.drawPile.back().id == CardId::BASH && sampled.cards.drawPile.back().upgraded,
              "known top must survive resampling");
        check(sampled.cards.drawPile.front().id == CardId::DEFEND_RED && sampled.cards.drawPile.front().upgraded,
              "known bottom must survive resampling");
    }
    changed = known; std::swap(changed.cards.drawPile.back(), changed.cards.drawPile[1]);
    check(public_search_key(changed) != public_search_key(known), "known top must distinguish nodes");
    const auto trajectoryRequest = [](const BattleContext &state, int count, std::uint64_t seed) {
        PublicBeliefCombatSearch sampled(std::vector<BattleContext>(count, state), seed, 2);
        sampled.setObjective(1, 35, 4, 0, 0); sampled.enablePolicyPriors(1.5, .15);
        const auto ids = sampled.requestBatch(1, 1, 0, 0);
        check(ids.size() == 1, "root evaluation request missing");
        return sampled.pending.at(ids.front()).state;
    };
    const auto sampledKnown = trajectoryRequest(known, 2, 42);
    check(publicSearchWords(sampledKnown) == publicSearchWords(known),
          "trajectory rejuvenation must preserve the public root");
    check(sampledKnown.cards.drawPile.back().id == CardId::BASH
          && sampledKnown.cards.drawPile.front().id == CardId::DEFEND_RED,
          "trajectory rejuvenation must preserve known draw ends");
    const auto sampledAgain = trajectoryRequest(known, 2, 43);
    check(sampledKnown.aiRng.seed0 != sampledAgain.aiRng.seed0,
          "different trajectories must sample different hidden futures");
    const auto sampledEye = trajectoryRequest(eye, 2, 42);
    for (std::size_t i = 0; i < eye.cards.drawPile.size(); ++i)
        check(sampledEye.cards.drawPile[i].id == eye.cards.drawPile[i].id,
              "trajectory sampling must preserve Frozen Eye order");
    const auto deterministic = trajectoryRequest(known, 1, 42);
    check(deterministic.aiRng.seed0 == known.aiRng.seed0,
          "single-particle deterministic API must keep its hidden future");
    std::cout << "fresh public trajectories, known ends, Frozen Eye and deterministic API PASS\n";

    known.cards.popFromDrawPile();
    check(known.cards.knownDrawTop == 0, "drawing must consume top knowledge");
    known.cards.removeFromDrawPileAtIdx(0);
    check(known.cards.knownDrawBottom == 0, "removal must consume bottom knowledge");

    Counts slots, semantics;
    enumerate(root, true, slots); enumerate(root, false, semantics);
    std::cout << "turn-one card-play prefixes: slots=" << slots.prefixes << ", semantic="
        << semantics.prefixes << ", fused=" << semantics.unique.size() << '\n';
    check(slots.prefixes == 50 && semantics.prefixes == 15 && semantics.unique.size() == 11,
          "starter-turn reduction changed unexpectedly");

    std::vector<BattleContext> particles;
    for (int i = 0; i < 8; ++i) {
        auto state = root; PublicBeliefCombatSearch::resampleDraw(state, root, 100 + i);
        particles.push_back(std::move(state));
    }
    PublicBeliefCombatSearch tree(particles, 42, 2);
    tree.maximumActions = 3;
    while (tree.simulations < 1000) {
        for (const auto id : tree.requestBatch(1, 1000, 0, 0)) tree.submit(id, .5);
    }
    const auto &strike = cardEdge(tree.root(), root, CardId::STRIKE_RED);
    const auto &defend = cardEdge(tree.root(), root, CardId::DEFEND_RED);
    check(strike.children.size() == 1 && defend.children.size() == 1, "ordinary plays have one public successor");
    const auto &strikeNode = *tree.nodes.at(*strike.children.begin());
    const auto &defendNode = *tree.nodes.at(*defend.children.begin());
    const auto &strikeDefend = cardEdge(strikeNode, afterStrike, CardId::DEFEND_RED);
    const auto &defendStrike = cardEdge(defendNode, afterDefend, CardId::STRIKE_RED);
    check(!strikeDefend.children.empty() && strikeDefend.children == defendStrike.children,
          "actual DAG must share commuting continuations");
    const auto reusedKey = strikeNode.key;
    const auto playedSemantic = strike.semanticKey;
    auto advanced = tree.advanceParticles(playedSemantic, afterStrike);
    check(advanced.size() == 8 && advanced.front().aiRng.seed0 == particles.front().aiRng.seed0,
          "ordinary root advancement must propagate, not reseed particles");
    tree.rebase(advanced, playedSemantic, 123);
    check(tree.root().key == reusedKey && tree.retainedVisits > 0 && tree.retainedVisits <= 256,
          "root advancement must preserve bounded continuation evidence");
    const auto nextSemantic = cardEdge(tree.root(), afterStrike, CardId::DEFEND_RED).semanticKey;
    const auto commonKey = *cardEdge(tree.root(), afterStrike, CardId::DEFEND_RED).children.begin();
    for (auto &state : advanced) { state = play(std::move(state), CardId::DEFEND_RED); state.shuffleRng.random(); }
    tree.rebase(advanced, nextSemantic, 124);
    check(tree.root().key == commonKey && tree.retainedVisits > 0,
          "fresh hidden streams must not erase matching public evidence");
    std::cout << "shared DAG and warm root advancement PASS: nodes=" << tree.nodes.size()
        << ", retained_visits=" << tree.retainedVisits << ", hits=" << tree.transpositionHits << '\n';

    // A first-visit rollout records only a successor key. If another route later
    // searches that state, root pruning must still retain the now-useful evidence.
    PublicBeliefCombatSearch links(particles, 42, 2);
    links.maximumActions = 4; links.exploration = 0;
    auto &linkRoot = *links.nodes.at(links.root().key);
    const auto defendSemantic = semantic_action_key(root, playAction(root, CardId::DEFEND_RED));
    for (auto &edge : linkRoot.edges) { edge.visits = 1; edge.valueSum = edge.semanticKey == defendSemantic ? 1. : 0.; }
    linkRoot.visits = linkRoot.edges.size();
    std::vector<BattleContext> defendStates, commonStates;
    for (const auto &state : particles) {
        defendStates.push_back(play(state, CardId::DEFEND_RED));
        commonStates.push_back(play(defendStates.back(), CardId::STRIKE_RED));
    }
    PublicBeliefCombatSearch defendBranch(defendStates, 43, 2);
    auto &defendRoot = *defendBranch.nodes.at(defendBranch.root().key);
    const auto followStrike = semantic_action_key(afterDefend, playAction(afterDefend, CardId::STRIKE_RED));
    for (auto &edge : defendRoot.edges) edge.visits = edge.semanticKey == followStrike ? 0 : 1;
    defendRoot.visits = defendRoot.edges.size() - 1;
    links.nodes.merge(defendBranch.nodes);
    for (const auto id : links.requestBatch(1, 1, 0, 0)) links.submit(id, .5);
    PublicBeliefCombatSearch commonBranch(commonStates, 44, 2);
    commonBranch.maximumActions = 1;
    while (commonBranch.simulations < 8)
        for (const auto id : commonBranch.requestBatch(1, 8, 0, 0)) commonBranch.submit(id, .5);
    const auto laterKnownKey = commonBranch.root().key;
    links.nodes.merge(commonBranch.nodes);
    links.rebase(defendStates, defendSemantic, 45);
    check(links.nodes.count(laterKnownKey) && links.nodes.at(laterKnownKey)->visits == 8,
          "deferred leaf link must preserve evidence later learned through another route");
    std::cout << "deferred leaf links retain later shared evidence PASS\n";

    // Deep boss budgets must not be reduced to a toy 256-visit warm start.
    PublicBeliefCombatSearch deep(particles, 42, 2);
    deep.maximumActions = 4;
    while (deep.simulations < 4096)
        for (const auto id : deep.requestBatch(1, 4096, 0, 0)) deep.submit(id, .5);
    const auto strikeSemantic = cardEdge(deep.root(), root, CardId::STRIKE_RED).semanticKey;
    auto deepStates = deep.advanceParticles(strikeSemantic, afterStrike);
    deep.rebase(deepStates, strikeSemantic, 42);
    check(deep.retainedVisits > 256 && deep.retainedVisits <= 4096,
          "deep search confidence must scale with actual compute, not a fixed 256 cap");
    std::cout << "deep-budget warm confidence PASS: " << deep.retainedVisits << '\n';

    PublicBeliefCombatSearch guided(particles, 42, 2);
    guided.maximumActions = 128; guided.search(200);
    check(guided.simulations == 200 && guided.selectedAction().isValidAction(guided.particles.front()),
          "multi-turn guided search must return a legal root action");

    PublicBeliefCombatSearch pv(particles, 42, 2);
    pv.maximumActions = 4; pv.setObjective(1, 35, 4, 0, 0); pv.enablePolicyPriors(1.5, .15);
    while (pv.simulations < 100) {
        for (const auto id : pv.requestBatch(4, 100, 0, 0)) {
            const auto &request = pv.pending.at(id);
            std::vector<double> priors;
            if (request.child) priors.assign(request.child->edges.size(), 1.);
            pv.submit(id, .5, priors);
        }
    }
    check(pv.selectedAction().isValidAction(pv.particles.front()), "PUCT root action must be legal");
    std::cout << "guided and batched PUCT smoke PASS\n";
}
