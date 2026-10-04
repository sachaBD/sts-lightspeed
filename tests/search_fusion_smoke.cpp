#include "game/GameContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <chrono>
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
    BattleContext state;
    state.init(game, MonsterEncounter::JAW_WORM);
    check(state.turn == 0 && isSettledSearchState(state), "fixture must start at turn one decision one");
    return state;
}
search::Action playAction(const BattleContext &s, CardId card) {
    for (const auto action : search::Action::getAllActionsInState(s))
        if (action.getActionType() == ActionType::CARD && s.cards.hand[action.getSourceIdx()].id == card)
            return action;
    throw std::runtime_error("fixture card not playable");
}
BattleContext play(BattleContext s, CardId card) {
    playAction(s, card).execute(s); canonicalizeSearchState(s); return s;
}
struct Counts {
    std::size_t slotPrefixes = 0, semanticPrefixes = 0;
    std::set<std::vector<std::uint64_t>> unique;
};
void enumerate(BattleContext state, bool slots, Counts &counts) {
    if (slots) ++counts.slotPrefixes; else ++counts.semanticPrefixes;
    canonicalizeSearchState(state);
    counts.unique.insert(searchStateWords(state));
    std::set<std::uint64_t> actions;
    for (const auto action : search::Action::getAllActionsInState(state)) {
        if (action.getActionType() != ActionType::CARD) continue;
        if (!slots && !actions.insert(semantic_action_key(state, action)).second) continue;
        auto child = state;
        action.execute(child);
        check(child.turn == 0 && isSettledSearchState(child), "unexpected turn/selection in starter-deck enumeration");
        enumerate(std::move(child), slots, counts);
    }
}
const PublicBeliefCombatSearch::Edge &cardEdge(const PublicBeliefCombatSearch::Node &node, CardId id) {
    for (const auto &edge : node.edges)
        if (edge.action.getActionType() == ActionType::CARD
            && node.belief.front().state.cards.hand[edge.action.getSourceIdx()].id == id) return edge;
    throw std::runtime_error("card edge missing");
}
}
int main() {
    auto original = jawWorm(0);
    // Find one natural starter hand containing Strike, Defend and Bash; no deck edits.
    std::uint64_t seed = 0;
    for (; seed < 100; ++seed) {
        original = jawWorm(seed);
        bool strike = false, defend = false, bash = false;
        for (int i = 0; i < original.cards.cardsInHand; ++i) {
            strike |= original.cards.hand[i].id == CardId::STRIKE_RED;
            defend |= original.cards.hand[i].id == CardId::DEFEND_RED;
            bash |= original.cards.hand[i].id == CardId::BASH;
        }
        if (strike && defend && bash) break;
    }
    check(seed < 100, "no suitable natural starter hand");
    auto root = original; canonicalizeSearchState(root);
    std::cout << "fixture: Jaw Worm, default Ironclad deck, A0, seed=" << seed << ", turn=" << root.turn + 1 << " (native zero-based), energy=" << root.player.energy << ", hand=";
    for (int i = 0; i < root.cards.cardsInHand; ++i) std::cout << root.cards.hand[i].getName() << ',';
    std::cout << '\n';
    const auto sd = play(play(root, CardId::STRIKE_RED), CardId::DEFEND_RED);
    const auto ds = play(play(root, CardId::DEFEND_RED), CardId::STRIKE_RED);
    check(searchStateWords(sd) == searchStateWords(ds), "Strike/Defend must commute");
    check(searchStateWords(play(play(root, CardId::BASH), CardId::STRIKE_RED))
        != searchStateWords(play(play(root, CardId::STRIKE_RED), CardId::BASH)), "Bash/Strike must not commute");
    auto permuted = root;
    std::swap(permuted.cards.hand[0], permuted.cards.hand[1]);
    check(public_search_key(permuted) == public_search_key(root), "hand order is not public search identity");
    canonicalizeSearchState(permuted);
    check(searchStateWords(permuted) == searchStateWords(root), "hand fingerprint must be unordered");
    const auto source = playAction(root, CardId::STRIKE_RED);
    const auto mapped = mapSemanticAction(root, source, original);
    check(original.cards.hand[mapped.getSourceIdx()].id == CardId::STRIKE_RED, "root action slot remapping");
    auto changed = root; std::swap(changed.cards.drawPile[0], changed.cards.drawPile[1]);
    check(searchStateWords(changed) != searchStateWords(root), "draw order must survive fingerprinting");
    changed = root; changed.cardRandomRng.random();
    check(searchStateWords(changed) != searchStateWords(root), "RNG state must survive fingerprinting");
    changed = root; changed.player.stance = Stance::WRATH;
    check(public_search_key(changed) != public_search_key(root), "stance must be observable");
    changed = root; ++changed.player.echoFormCardsDoubled;
    check(searchStateWords(changed) != searchStateWords(root), "trigger counters must survive fingerprinting");
    changed = root; ++changed.monsters.arr[0].miscInfo;
    check(searchStateWords(changed) != searchStateWords(root), "hidden monster state must survive fingerprinting");
    Counts slots, semantics;
    enumerate(root, true, slots); enumerate(root, false, semantics);
    std::cout << "exhaustive turn-one card-play prefixes (including root, excluding End Turn): slots="
        << slots.slotPrefixes << ", semantic=" << semantics.semanticPrefixes
        << ", fused=" << semantics.unique.size() << '\n';
    check(semantics.unique.size() < semantics.semanticPrefixes, "must reduce play-order duplicates beyond slot deduplication");

    std::vector<BattleContext> particles;
    for (int i = 0; i < 8; ++i) {
        auto state = root;
        PublicBeliefCombatSearch::resampleDraw(state, root, 100 + i);
        particles.push_back(std::move(state));
    }
    PublicBeliefCombatSearch tree(particles, 42, 2);
    tree.maximumActions = 3;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        const auto ids = tree.requestBatch(1, 1000, 0, 0);
        for (const auto id : ids) tree.submit(id, .5);
    }
    const auto &strike = cardEdge(tree.root(), CardId::STRIKE_RED);
    const auto &defend = cardEdge(tree.root(), CardId::DEFEND_RED);
    check(!strike.successors.empty() && !defend.successors.empty(), "root edges must be explored");
    const auto &afterStrike = *tree.nodes.at(strike.successors.front());
    const auto &afterDefend = *tree.nodes.at(defend.successors.front());
    const auto &strikeDefend = cardEdge(afterStrike, CardId::DEFEND_RED);
    const auto &defendStrike = cardEdge(afterDefend, CardId::STRIKE_RED);
    check(!strikeDefend.successors.empty() && !defendStrike.successors.empty(), "commuting continuations must be explored");
    check(strikeDefend.successors == defendStrike.successors, "actual DAG must fuse Strike/Defend for every particle");
    check(tree.nodes.at(strikeDefend.successors.front())->belief.size() == 8, "fusion must preserve the full hidden cohort");
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "8-particle bounded smoke: simulations=" << tree.simulations << ", nodes=" << tree.nodes.size()
        << ", transposition_hits=" << tree.transpositionHits << ", expanded_transitions=" << tree.expandedTransitions
        << ", cached_particle_states=" << tree.cachedParticleStates << ", BattleContext_bytes=" << sizeof(BattleContext)
        << ", seconds=" << elapsed << '\n';
    // Reuse only exact beliefs, never merely the same visible state after resampling.
    const auto retained = afterStrike.visits;
    const auto playedSemantic = strike.semanticKey;
    std::vector<BattleContext> advanced;
    for (const auto &particle : afterStrike.belief) advanced.push_back(particle.state);
    tree.rebase(advanced, playedSemantic, 123);
    check(tree.retainedVisits == retained && retained > 0, "exact-belief rebase must retain visits");
    const auto nextSemantic = cardEdge(tree.root(), CardId::DEFEND_RED).semanticKey;
    for (auto &state : advanced) {
        state = play(std::move(state), CardId::DEFEND_RED);
        state.shuffleRng.random();
    }
    tree.rebase(advanced, nextSemantic, 124);
    check(tree.retainedVisits == 0 && tree.nodes.size() == 1, "changed hidden belief must not reuse cached evidence");

    PublicBeliefCombatSearch guided(particles, 42, 2);
    guided.maximumActions = 128;
    guided.search(200);
    check(guided.simulations == 200 && guided.selectedAction().isValidAction(guided.particles.front()),
          "multi-turn guided search must return a legal root action");

    PublicBeliefCombatSearch pv(particles, 42, 2);
    pv.maximumActions = 4;
    pv.setObjective(1, 35, 4, 0, 0);
    pv.enablePolicyPriors(1.5, .15);
    while (pv.simulations < 100) {
        const auto ids = pv.requestBatch(4, 100, 0, 0);
        for (const auto id : ids) {
            const auto &request = pv.pending.at(id);
            std::vector<double> priors;
            if (request.child) priors.assign(request.child->edges.size(), 1.);
            pv.submit(id, .5, priors);
        }
    }
    check(pv.selectedAction().isValidAction(pv.particles.front()), "PUCT DAG root action must be legal");
    std::cout << "PASS: actual Strike->Defend and Defend->Strike successors are the same shared belief node\n";
}
