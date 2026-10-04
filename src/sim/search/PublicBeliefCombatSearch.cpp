#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace sts::search {
namespace {
constexpr std::uint64_t ROOT_KEY = 0xcbf29ce484222325ULL;
void append(std::uint64_t &hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (i * 8)) & 255;
        hash *= 0x100000001b3ULL;
    }
}
std::uint64_t cardKey(const CardInstance &card) {
    std::uint64_t key = ROOT_KEY;
    // UUID/uniqueId and the originating deck/shuffle index are not features.
    append(key, static_cast<int>(card.id)); append(key, card.getUpgradeCount());
    append(key, card.specialData); append(key, card.cost);
    append(key, card.costForTurn); append(key, card.freeToPlayOnce);
    append(key, card.retain);
    return key;
}
// Validates and normalizes priors onto edges (in edge order).
template<class Edges> void assignPriors(Edges &edges, const std::vector<double> &priors) {
    if (priors.size() != edges.size()) throw std::invalid_argument("prior count differs from edge count");
    double sum = 0;
    for (double p : priors) {
        if (!std::isfinite(p) || p < 0) throw std::invalid_argument("invalid prior probability");
        sum += p;
    }
    if (!(sum > 0)) throw std::invalid_argument("empty prior support");
    for (std::size_t i = 0; i < edges.size(); ++i) edges[i].prior = priors[i] / sum;
}

}

std::uint64_t PublicBeliefCombatSearch::publicActionKey(const BattleContext &state, Action action) {
    return semantic_action_key(state, action);
}
std::uint64_t PublicBeliefCombatSearch::actionKey(const BattleContext &state, Action action) const {
    return semantic_action_key(state, action);
}
std::uint64_t PublicBeliefCombatSearch::identityActionKey(const BattleContext &state, Action action) {
    return semantic_action_key(state, action);
}
Action PublicBeliefCombatSearch::mapAction(const BattleContext &source, Action action,
                                          const BattleContext &target) {
    return mapSemanticAction(source, action, target);
}

void PublicBeliefCombatSearch::resampleDraw(BattleContext &target,
                                           const BattleContext &observed, std::uint64_t seed) {
    auto &cards = target.cards.drawPile;
    cards = observed.cards.drawPile;
    if (observed.player.hasRelic<RelicId::FROZEN_EYE>()) return;
    auto begin = cards.begin();
    // The native Scry menu indexes its exposed prefix. Preserve that public
    // menu, never the rest of the hidden pile. Other card-selection screens
    // do not make draw order public; their actions are matched by card identity.
    if (observed.inputState == InputState::CARD_SELECT
        && observed.cardSelectInfo.cardSelectTask == CardSelectTask::SCRY) {
        begin += std::min(observed.cardSelectInfo.pickCount, static_cast<int>(cards.size()));
    }
    std::sort(begin, cards.end(), [](const auto &a, const auto &b) { return cardKey(a) < cardKey(b); });
    std::default_random_engine engine(seed);
    std::shuffle(begin, cards.end(), engine);
}

std::uint64_t PublicBeliefCombatSearch::publicObservation(const BattleContext &state) {
    return public_search_key(state);
}
std::uint64_t PublicBeliefCombatSearch::observationKey(const BattleContext &state) {
    return public_search_key(state);
}

PublicBeliefCombatSearch::PublicBeliefCombatSearch(
    std::vector<BattleContext> states, std::uint64_t seed, int rolloutMode)
    : particles(std::move(states)), random(seed), rollout(particles.at(0)),
      normalization(100.0 * (35 + particles.at(0).player.maxHp + 20)) {
    normalizationMaxHp = particles.front().player.maxHp;
    rollout.rolloutMode = rolloutMode;
    rollout.randGen.seed(seed);
    std::vector<Particle> belief;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        canonicalizeSearchState(particles[i]);
        belief.push_back({static_cast<int>(i), particles[i]});
    }
    rootKey = node(std::move(belief), ROOT_KEY, ROOT_KEY).key;
}

std::vector<std::uint64_t> PublicBeliefCombatSearch::signatureWords(
    const std::vector<Particle> &belief, std::uint64_t scope, std::uint64_t history) {
    const bool settled = std::all_of(belief.begin(), belief.end(),
                                    [](const Particle &p) { return isSettledSearchState(p.state); });
    std::vector<std::uint64_t> signature{settled ? 0x5354415445ULL : 0x484953544f5259ULL, scope};
    if (settled) {
        signature.reserve(3 + belief.size() * 512);
        signature.push_back(belief.size());
        for (const auto &particle : belief) {
            const auto words = searchStateWords(particle.state);
            signature.push_back(particle.id); signature.push_back(words.size());
            signature.insert(signature.end(), words.begin(), words.end());
        }
    } else signature.push_back(history); // Opaque pending action queues are not fused.
    return signature;
}

PublicBeliefCombatSearch::Node &PublicBeliefCombatSearch::node(
    std::vector<Particle> belief, std::uint64_t scope, std::uint64_t history) {
    if (belief.empty()) throw std::logic_error("empty public belief");
    const auto observation = publicSearchWords(belief.front().state);
    const auto signature = signatureWords(belief, scope, history);
    const auto key = hashSearchWords(signature);
    if (const auto found = nodes.find(key); found != nodes.end()) {
        const auto &existing = *found->second;
        if (signatureWords(existing.belief, existing.scope, existing.history) != signature)
            throw std::runtime_error("public-belief node hash collision");
        if (isSettledSearchState(belief.front().state)) ++transpositionHits;
        return *found->second;
    }
    auto created = std::make_unique<Node>();
    created->key = key; created->scope = scope; created->history = history;
    created->belief = std::move(belief);
    const auto &representative = created->belief.front().state;
    std::vector<std::uint64_t> menu;
    if (representative.outcome == Outcome::UNDECIDED) {
        BattleScumSearcher2::Node temporary;
        rollout.enumerateActionsForNode(temporary, representative);
        for (const auto &edge : temporary.edges) {
            if (!edge.action.isValidAction(representative))
                throw std::runtime_error("native enumeration produced an illegal belief action");
            const auto semantic = actionKey(representative, edge.action);
            if (std::find(menu.begin(), menu.end(), semantic) == menu.end()) {
                menu.push_back(semantic); created->edges.push_back({edge.action, semantic});
            }
        }
        if (menu.empty()) throw std::runtime_error("empty belief-search legal menu");
    }
    std::sort(menu.begin(), menu.end());
    for (std::size_t i = 1; i < created->belief.size(); ++i) {
        const auto &particle = created->belief[i];
        if (publicSearchWords(particle.state) != observation)
            throw std::runtime_error("cohort mixes distinct public observations");
        std::vector<std::uint64_t> other;
        if (particle.state.outcome == Outcome::UNDECIDED) {
            BattleScumSearcher2::Node temporary;
            rollout.enumerateActionsForNode(temporary, particle.state);
            for (const auto &edge : temporary.edges) {
                if (!edge.action.isValidAction(particle.state))
                    throw std::runtime_error("illegal particle action");
                other.push_back(actionKey(particle.state, edge.action));
            }
        }
        std::sort(other.begin(), other.end());
        other.erase(std::unique(other.begin(), other.end()), other.end());
        if (menu != other) throw std::runtime_error("public observation aliases incompatible legal menus");
    }
    std::shuffle(created->edges.begin(), created->edges.end(), random);
    if (policyPriors)
        for (auto &edge : created->edges) edge.prior = 1.0 / created->edges.size();
    cachedParticleStates += created->belief.size();
    auto *result = created.get(); nodes.emplace(key, std::move(created));
    return *result;
}

std::size_t PublicBeliefCombatSearch::particleIndex(const Node &at, int id) {
    for (std::size_t i = 0; i < at.belief.size(); ++i) if (at.belief[i].id == id) return i;
    throw std::logic_error("sampled particle missing from belief cohort");
}

void PublicBeliefCombatSearch::expand(Node &parent, std::size_t choice) {
    auto &edge = parent.edges.at(choice);
    if (!edge.successors.empty()) return;
    struct Group { std::vector<Particle> belief; std::vector<std::uint64_t> observation; };
    std::map<std::uint64_t, Group> groups;
    std::vector<std::uint64_t> observations;
    for (const auto &particle : parent.belief) {
        auto state = particle.state;
        const auto action = mapAction(parent.belief.front().state, edge.action, state);
        action.execute(state);
        if (state.unsupportedEffectKind != UnsupportedEffectKind::NONE)
            throw std::runtime_error("unsupported effect in public-belief transition");
        canonicalizeSearchState(state);
        const auto words = publicSearchWords(state);
        const auto observation = hashSearchWords(words);
        auto &group = groups[observation];
        if (!group.belief.empty() && group.observation != words)
            throw std::runtime_error("public observation hash collision");
        if (group.belief.empty()) group.observation = words;
        group.belief.push_back({particle.id, std::move(state)});
        observations.push_back(observation);
    }
    std::map<std::uint64_t, std::uint64_t> successors;
    for (auto &[observation, group] : groups) {
        auto history = parent.key;
        append(history, edge.semanticKey); append(history, observation);
        const auto scope = group.belief.front().state.turn == parent.belief.front().state.turn
                           ? parent.scope : history;
        auto &child = node(std::move(group.belief), scope, history);
        parent.children.insert(child.key); successors[observation] = child.key;
    }
    for (const auto observation : observations) edge.successors.push_back(successors.at(observation));
    ++expandedTransitions;
}

std::size_t PublicBeliefCombatSearch::select(const Node &current) {
    double best = -std::numeric_limits<double>::infinity();
    std::size_t selected = 0;
    // Loop invariants hoisted (same arithmetic, so the same choices).
    const double logParent = std::log(current.visits + current.inFlight + 1.0);
    const bool usePrior = priorStrength > 0 && &current == &root();
    const double sqrtParent = usePrior ? std::sqrt(current.visits + current.inFlight + 1.0) : 0.0;
    for (std::size_t i = 0; i < current.edges.size(); ++i) {
        const auto &edge = current.edges[i];
        const auto visits = edge.visits + edge.inFlight;
        if (visits == 0) return i;
        const double value = maxBackup && edge.visits ? edge.best : edge.valueSum / visits;
        auto score = value
            + exploration * std::sqrt(logParent / visits);
        if (usePrior)
            score += priorStrength * edge.prior * sqrtParent / (1 + visits);
        if (score > best) { best = score; selected = i; }
    }
    return selected;
}

// PUCT with first-play urgency. fpuBase: this node's mean backed-up value, else its own leaf value,
// else (evaluation still pending) the parent's base, passed in. Unvisited edges score Q = base -
// fpuReduction; in-flight edges keep the historical virtual loss Q = valueSum / (visits + inFlight).
std::size_t PublicBeliefCombatSearch::selectPuct(const Node &current, double &fpuBase) const {
    if (current.visits > 0) {
        double sum = 0;
        for (const auto &edge : current.edges) sum += edge.valueSum;
        fpuBase = sum / current.visits;
    } else if (current.evaluated) {
        fpuBase = current.value;
    }
    const double fpu = fpuBase - fpuReduction;
    const double sqrtParent = std::sqrt(current.visits + current.inFlight + 1.0);
    double best = -std::numeric_limits<double>::infinity();
    std::size_t selected = 0;
    for (std::size_t i = 0; i < current.edges.size(); ++i) {
        const auto &edge = current.edges[i];
        const auto visits = edge.visits + edge.inFlight;
        const double value = visits == 0 ? fpu : maxBackup && edge.visits ? edge.best : edge.valueSum / visits;
        const double score = value + cPuct * edge.prior * sqrtParent / (1 + visits);
        if (score > best) { best = score; selected = i; }
    }
    return selected;
}

void PublicBeliefCombatSearch::addRequest(const BattleContext &state, Path &&path, Node *child) {
    // Reuse a submitted request's map node: no allocation, and copy-assignment keeps the state's
    // pile capacity.
    Request *slot;
    if (spareRequests.empty()) {
        slot = &pending.emplace_hint(pending.end(), std::piecewise_construct,
                                     std::forward_as_tuple(++nextRequest), std::forward_as_tuple())->second;
    } else {
        auto handle = std::move(spareRequests.back());
        spareRequests.pop_back();
        handle.key() = ++nextRequest;
        slot = &pending.insert(pending.end(), std::move(handle))->second;
    }
    slot->state = state; slot->path = std::move(path); slot->child = child;
}

void PublicBeliefCombatSearch::simulate(int particle, bool request, int rolloutTurns, int rolloutSteps) {
    auto *at = nodes.at(rootKey).get();
    Path path;
    if (policyPriors && !rootRequested) {
        rootRequested = true;
        addRequest(at->belief.at(particleIndex(*at, particle)).state, std::move(path), at);
        return;
    }
    Node *leaf = nullptr;
    double fpuBase = 0;
    BattleContext current = at->belief.at(particleIndex(*at, particle)).state;
    for (int depth = 0; depth < maximumActions && current.outcome == Outcome::UNDECIDED; ++depth) {
        if (std::any_of(path.begin(), path.end(), [&](const auto &p) { return p.first == at; }))
            throw std::runtime_error("cycle in within-turn public-belief DAG");
        const auto choice = policyPriors ? selectPuct(*at, fpuBase) : select(*at);
        const bool unvisited = at->edges[choice].visits + at->edges[choice].inFlight == 0;
        ++at->inFlight; ++at->edges[choice].inFlight;
        path.emplace_back(at, choice);
        // UCB's first visit evaluates a sampled continuation, not a new policy node.
        // Build the full conditioned cohort only when tree traversal needs that child.
        if (unvisited && !policyPriors) {
            mapAction(at->belief.front().state, at->edges[choice].action, current).execute(current);
            canonicalizeSearchState(current);
            boundedRollout(current, request ? rolloutTurns : -1, request ? rolloutSteps : maximumActions);
            break;
        }
        const auto index = particleIndex(*at, particle);
        expand(*at, choice);
        at = nodes.at(at->edges[choice].successors.at(index)).get();
        current = at->belief.at(particleIndex(*at, particle)).state;
        if (policyPriors && current.outcome == Outcome::UNDECIDED && !at->evaluated) {
            leaf = at;
            break;
        }
    }
    if (current.unsupportedEffectKind != UnsupportedEffectKind::NONE)
        throw std::runtime_error("unsupported effect in unified public combat search");
    if (request && current.outcome == Outcome::UNDECIDED) {
        addRequest(current, std::move(path), leaf);
        return;
    }
    backup(path, terminalValue(current), &current);
}

double PublicBeliefCombatSearch::terminalValue(const BattleContext &state) const {
    if (state.outcome == Outcome::UNDECIDED) return 0;
    if (objectiveMode == 0)
        return std::clamp(BattleScumSearcher2::evaluateEndState(state) / normalization, -0.1, 2.0);
    const bool won = state.outcome == Outcome::PLAYER_VICTORY && state.player.curHp > 0 && !state.escapedCombat;
    if (!won) return 0;
    return scorePrediction(1., state.player.curHp, state.potionCount, state.player.maxHp, state.player.gold);
}

void PublicBeliefCombatSearch::backup(Path &path, double value, const BattleContext *terminal) {
    const bool resolved = terminal && terminal->outcome != Outcome::UNDECIDED;
    const bool won = resolved && terminal->outcome == Outcome::PLAYER_VICTORY
        && terminal->player.curHp > 0 && !terminal->escapedCombat;
    for (auto &entry : path) {
        --entry.first->inFlight;
        ++entry.first->visits;
        auto &edge = entry.first->edges[entry.second];
        --edge.inFlight;
        ++edge.visits; edge.valueSum += value; edge.wins += won; edge.best = std::max(edge.best, value);
        if (resolved) {
            ++edge.measured;
            if (won) {
                edge.outcomeSum[0] += 1.; edge.outcomeSum[1] += terminal->player.curHp;
                edge.outcomeSum[2] += terminal->potionCount; edge.outcomeSum[3] += terminal->player.maxHp;
                edge.outcomeSum[4] += terminal->player.gold;
            }
        }
    }
    ++simulations;
    if (resolved) ++terminalEvaluations;
    else if (terminal) ++unresolvedEvaluations;
    else ++neuralEvaluations;
}

void PublicBeliefCombatSearch::boundedRollout(BattleContext &state, int turns, int steps) {
    BattleScumSearcher2::Node temporary;
    const auto startTurn = state.turn;
    for (int step = 0; step < std::min(steps, maximumActions) && state.outcome == Outcome::UNDECIDED; ++step) {
        if (turns >= 0 && state.turn - startTurn >= turns) break;
        // Same move and draws as enumerate + selectRolloutAction; enumerates only when needed.
        rollout.rolloutAction(temporary, state).execute(state);
        canonicalizeSearchState(state);
    }
    // An unfinished rollout is unresolved evidence, never a victory.
}

void PublicBeliefCombatSearch::search(std::int64_t budget) {
    if (budget <= 0 || !pending.empty()) throw std::invalid_argument("invalid or pending public search budget");
    if (policyPriors) throw std::logic_error("policy-prior search needs leaf evaluations (requestBatch)");
    // Draw independently from the root belief. Cycling deterministically can
    // phase-lock the UCT action schedule to particle identity and bias values.
    std::uniform_int_distribution<int> sample(0, static_cast<int>(particles.size()) - 1);
    for (std::int64_t i = 0; i < budget; ++i) simulate(particles.size() == 1 ? 0 : sample(random));
}

const PublicBeliefCombatSearch::Node &PublicBeliefCombatSearch::root() const {
    return *nodes.at(rootKey);
}

Action PublicBeliefCombatSearch::selectedAction() const {
    if (!pending.empty()) throw std::logic_error("cannot select an action with pending leaves");
    const auto &edges = root().edges;
    auto best = std::max_element(edges.begin(), edges.end(), [this](const Edge &a, const Edge &b) {
        if (maxBackup && a.visits && b.visits && a.best != b.best) return a.best < b.best;
        if (a.visits != b.visits) return a.visits < b.visits;
        return a.valueSum < b.valueSum;
    });
    if (best == edges.end() || best->visits == 0) throw std::runtime_error("empty belief search");
    return best->action;
}

void PublicBeliefCombatSearch::setRootPrior(const std::vector<double> &priors, double strength) {
    if (policyPriors) throw std::logic_error("root prior cannot be combined with policy-prior mode");
    if (!std::isfinite(strength) || strength < 0) throw std::invalid_argument("invalid unified root prior");
    assignPriors(nodes.at(rootKey)->edges, priors);
    priorStrength = strength;
}

void PublicBeliefCombatSearch::enablePolicyPriors(double puct, double reduction, double alpha, double epsilon,
                                                  std::uint64_t noiseSeed) {
    if (simulations || root().visits || !pending.empty())
        throw std::logic_error("policy-prior mode must be enabled before any evidence");
    if (objectiveMode != 1) throw std::logic_error("policy-prior mode requires objectiveMode 1");
    if (priorStrength > 0) throw std::logic_error("policy-prior mode cannot be combined with a root prior");
    if (!std::isfinite(puct) || puct <= 0 || !std::isfinite(reduction) || !std::isfinite(alpha) || alpha < 0
        || !std::isfinite(epsilon) || epsilon < 0 || epsilon > 1 || (epsilon > 0 && alpha <= 0))
        throw std::invalid_argument("invalid policy-prior parameters");
    policyPriors = true;
    cPuct = puct; fpuReduction = reduction; dirichletAlpha = alpha; dirichletEpsilon = epsilon;
    noiseRandom.seed(noiseSeed);
    rootRequested = false; rootNoised = false;
    for (auto &[key, n] : nodes) for (auto &edge : n->edges) edge.prior = 1.0 / n->edges.size();
}

// Root exploration noise: P = (1 - epsilon) P + epsilon Dirichlet(alpha), once per root, after the
// root's own priors are known.
void PublicBeliefCombatSearch::applyRootNoise() {
    auto &at = *nodes.at(rootKey);
    if (!policyPriors || dirichletEpsilon <= 0 || rootNoised || !at.evaluated) return;
    rootNoised = true;
    std::gamma_distribution<double> gamma(dirichletAlpha, 1.0);
    std::vector<double> noise(at.edges.size());
    double sum = 0;
    for (auto &x : noise) sum += x = gamma(noiseRandom);
    if (!(sum > 0)) return;
    for (std::size_t i = 0; i < noise.size(); ++i)
        at.edges[i].prior = (1 - dirichletEpsilon) * at.edges[i].prior + dirichletEpsilon * noise[i] / sum;
}

void PublicBeliefCombatSearch::setObjective(int mode, double victory, double potion, double maximumHp, double gold) {
    if (simulations || root().visits || !pending.empty() || mode < 0 || mode > 1)
        throw std::invalid_argument("objective is immutable once evidence exists");
    for (double v : {victory, potion, maximumHp, gold})
        if (!std::isfinite(v) || v < 0) throw std::invalid_argument("invalid objective coefficient");
    if (policyPriors && mode != 1) throw std::logic_error("policy-prior mode requires objectiveMode 1");
    objectiveMode = mode; victoryHp = victory; potionHp = potion; maxHpPrice = maximumHp; goldHpPrice = gold;
    normalizationMaxHp = particles.front().player.maxHp;
    if (mode == 0) normalization = 100.0 * (35 + normalizationMaxHp + 20);  // as constructed
    if (mode == 1) normalization = 100 * (victoryHp + particles.front().player.maxHp + 5*potionHp
                                       + maxHpPrice*particles.front().player.maxHp + goldHpPrice*particles.front().player.gold + 1);
}

double PublicBeliefCombatSearch::scorePrediction(double win, double hp, double potions, double maximumHp, double gold) const {
    if (!std::isfinite(win) || win < 0 || win > 1 || !std::isfinite(hp) || hp < 0
        || !std::isfinite(potions) || potions < 0 || !std::isfinite(maximumHp) || maximumHp < 0
        || !std::isfinite(gold) || gold < 0) throw std::invalid_argument("invalid predicted terminal resources");
    // HP/potions/maxHP/gold are unconditional surviving expectations, not
    // conditional means that would accidentally multiply P(win) twice.
    return std::clamp(100 * ((objectiveMode ? victoryHp : 35.)*win + hp
        + (objectiveMode ? potionHp : 4.)*potions + (objectiveMode ? maxHpPrice : 0.)*maximumHp
        + (objectiveMode ? goldHpPrice : 0.)*gold) / normalization, 0., 2.);
}

std::vector<std::uint64_t> PublicBeliefCombatSearch::requestBatch(int count, std::int64_t budget,
                                                               int turns, int steps) {
    if (count < 1 || budget <= simulations || turns < 0 || steps < 0 || !pending.empty())
        throw std::invalid_argument("invalid batched leaf request");
    if (policyPriors && (turns != 0 || steps != 0))
        throw std::invalid_argument("policy-prior mode evaluates the leaf itself: no rollout bounds");
    const auto first = nextRequest;
    std::uniform_int_distribution<int> sample(0, static_cast<int>(particles.size())-1);
    for (int n = 0; n < count && simulations + static_cast<std::int64_t>(pending.size()) < budget; ++n) {
        simulate(particles.size() == 1 ? 0 : sample(random), true, turns, steps);
        if (policyPriors && !nodes.at(rootKey)->evaluated) break;  // nothing to steer by before the root's priors
    }
    std::vector<std::uint64_t> result;
    for (const auto &[id, request] : pending) if (id > first) result.push_back(id);
    return result;
}

void PublicBeliefCombatSearch::submit(std::uint64_t id, double value) {
    submit(id, value, {});
}

void PublicBeliefCombatSearch::submit(std::uint64_t id, double value, const std::vector<double> &priors) {
    auto found = pending.find(id);
    if (found == pending.end() || !std::isfinite(value) || value < -.1 || value > 2)
        throw std::invalid_argument("invalid unified leaf submission");
    auto *child = found->second.child;
    if (!priors.empty() && !child) throw std::invalid_argument("priors for a request without a leaf node");
    if (child && !child->evaluated) {  // first evaluation of a node wins
        if (!priors.empty()) assignPriors(child->edges, priors);
        child->evaluated = true; child->value = value;
        if (child == nodes.at(rootKey).get()) applyRootNoise();
    }
    backup(found->second.path, value, nullptr);
    spareRequests.push_back(pending.extract(found));
}

void PublicBeliefCombatSearch::submitGuided(std::uint64_t id) {
    auto found = pending.find(id);
    if (found == pending.end()) throw std::invalid_argument("unknown guided leaf request");
    auto &state = found->second.state;
    boundedRollout(state, -1, maximumActions);
    if (state.unsupportedEffectKind != UnsupportedEffectKind::NONE)
        throw std::runtime_error("unsupported effect in guided leaf");
    const auto value = terminalValue(state);
    if (auto *child = found->second.child; child && !child->evaluated) {  // keeps uniform priors
        child->evaluated = true; child->value = value;
        if (child == nodes.at(rootKey).get()) applyRootNoise();
    }
    backup(found->second.path, value, &state);
    spareRequests.push_back(pending.extract(found));
}

void PublicBeliefCombatSearch::rebase(std::vector<BattleContext> states, std::uint64_t action, std::uint64_t seed) {
    if (states.empty() || !pending.empty()) throw std::invalid_argument("cannot rebase empty/pending search");
    if (std::none_of(root().edges.begin(), root().edges.end(), [&](const Edge &e) {return e.semanticKey == action;}))
        throw std::invalid_argument("rebase action was not legal at previous public root");
    auto history = rootKey;
    append(history, action); append(history, observationKey(states.front()));
    const auto scope = states.front().turn == particles.front().turn ? root().scope : history;
    particles = std::move(states);
    random.seed(seed); rollout.randGen.seed(seed);
    std::vector<Particle> belief;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        canonicalizeSearchState(particles[i]);
        belief.push_back({static_cast<int>(i), particles[i]});
    }
    // Pending function queues cannot be structurally compared across a fresh root belief.
    if (!std::all_of(belief.begin(), belief.end(), [](const Particle &p) {return isSettledSearchState(p.state);})) {
        nodes.clear(); cachedParticleStates = 0;
    }
    rootKey = node(std::move(belief), scope, history).key;
    std::unordered_set<std::uint64_t> keep;
    std::vector<std::uint64_t> todo{rootKey};
    while (!todo.empty()) {
        auto key = todo.back(); todo.pop_back();
        auto found = nodes.find(key);
        if (found == nodes.end() || !keep.insert(key).second) continue;
        for (auto child : found->second->children) todo.push_back(child);
    }
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (!keep.count(it->first)) {
            cachedParticleStates -= it->second->belief.size();
            it = nodes.erase(it);
        } else ++it;
    }
    retainedNodes = nodes.size(); retainedVisits = root().visits;
    if (policyPriors) {  // a retained root keeps its priors; a new one is requested first
        rootRequested = root().evaluated; rootNoised = false;
        noiseRandom.seed(seed ^ 0x6e6f697365ULL);
        applyRootNoise();
    }
    simulations = terminalEvaluations = unresolvedEvaluations = neuralEvaluations = 0;
}
}
