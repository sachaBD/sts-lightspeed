#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <cmath>
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
    target.cards.knownDrawTop = observed.cards.knownDrawTop;
    target.cards.knownDrawBottom = observed.cards.knownDrawBottom;
    if (target.cards.knownDrawTop < 0 || target.cards.knownDrawBottom < 0
        || target.cards.knownDrawTop + target.cards.knownDrawBottom > static_cast<int>(cards.size()))
        throw std::logic_error("invalid draw knowledge during resampling");
    if (observed.player.hasRelic<RelicId::FROZEN_EYE>()) return;
    auto begin = cards.begin() + target.cards.knownDrawBottom;
    auto end = cards.end() - target.cards.knownDrawTop;
    if (observed.inputState == InputState::CARD_SELECT
        && observed.cardSelectInfo.cardSelectTask == CardSelectTask::SCRY)
        begin = cards.begin() + std::min(std::max(target.cards.knownDrawBottom,
            observed.cardSelectInfo.pickCount), static_cast<int>(cards.size()) - target.cards.knownDrawTop);
    std::sort(begin, end, [](const auto &a, const auto &b) { return cardKey(a) < cardKey(b); });
    std::default_random_engine engine(seed);
    std::shuffle(begin, end, engine);
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
    const auto publicState = publicSearchWords(particles.front());
    for (const auto &state : particles)
        if (publicSearchWords(state) != publicState)
            throw std::invalid_argument("root particles must have the same public observation");
    rootKey = node(particles.front(), ROOT_KEY).key;
    expand(*nodes.at(rootKey), particles.front());
}

void PublicBeliefCombatSearch::nodeWords(
    const BattleContext &state, std::uint64_t history) {
    auto &signature = signatureScratch;
    publicSearchWords(state, signature);
    // Preserve the existing path-dependent objective's public resource counters.
    signature.push_back(state.energyWasted); signature.push_back(state.cardsDrawn);
    signature.push_back(isSettledSearchState(state) ? 0x5354415445ULL : 0x484953544f5259ULL);
    if (!isSettledSearchState(state)) signature.push_back(history);
    // A pending function queue cannot be reconstructed from visible fields alone.
    // At settled decisions no opaque queues remain, so histories can be fused.
}

std::pair<std::uint64_t, PublicBeliefCombatSearch::Node *> PublicBeliefCombatSearch::findNode(
    const BattleContext &state, std::uint64_t history) {
    nodeWords(state, history);
    const auto &signature = signatureScratch;
    const auto key = hashSearchWords(signature);
    if (const auto found = nodes.find(key); found != nodes.end()) {
        if (found->second->signature != signature)
            throw std::runtime_error("public-search node hash collision");
        if (isSettledSearchState(state)) ++transpositionHits;
        return {key, found->second.get()};
    }
    return {key, nullptr};
}

PublicBeliefCombatSearch::Node &PublicBeliefCombatSearch::createNode(std::uint64_t key) {
    auto created = std::make_unique<Node>();
    created->key = key; created->signature = signatureScratch;
    const auto [found, inserted] = nodes.emplace(key, std::move(created));
    if (!inserted) throw std::logic_error("duplicate public-search node creation");
    return *found->second;
}

PublicBeliefCombatSearch::Node &PublicBeliefCombatSearch::node(
    const BattleContext &state, std::uint64_t history) {
    const auto [key, found] = findNode(state, history);
    return found ? *found : createNode(key);
}

void PublicBeliefCombatSearch::expand(Node &at, const BattleContext &state) {
    if (at.expanded) return;
    if (state.outcome == Outcome::UNDECIDED) {
        BattleScumSearcher2::Node temporary;
        rollout.enumerateActionsForNode(temporary, state);
        for (const auto &edge : temporary.edges) {
            if (!edge.action.isValidAction(state))
                throw std::runtime_error("native enumeration produced an illegal public action");
            const auto semantic = actionKey(state, edge.action);
            if (std::none_of(at.edges.begin(), at.edges.end(),
                            [&](const Edge &e) {return e.semanticKey == semantic;}))
                at.edges.push_back({edge.action, semantic});
        }
        if (at.edges.empty()) throw std::runtime_error("empty public-search legal menu");
    }
    std::shuffle(at.edges.begin(), at.edges.end(), random);
    if (policyPriors)
        for (auto &edge : at.edges) edge.prior = 1.0 / at.edges.size();
    at.expanded = true;
}

Action PublicBeliefCombatSearch::edgeAction(const Edge &edge, const BattleContext &state) const {
    const auto type = edge.action.getActionType();
    if (state.inputState == InputState::PLAYER_NORMAL) {
        if (type == ActionType::CARD || type == ActionType::POTION) {
            const auto count = type == ActionType::CARD ? state.cards.cardsInHand : state.potionCapacity;
            const auto original = edge.action.getSourceIdx();
            if (original < count && actionKey(state, edge.action) == edge.semanticKey
                && edge.action.isValidAction(state)) return edge.action;
            for (int i = 0; i < count; ++i) {
                if (i == original) continue;
                const Action action(type, i, edge.action.getTargetIdx());
                if (actionKey(state, action) == edge.semanticKey && action.isValidAction(state)) return action;
            }
        } else if (edge.action.isValidAction(state)
                   && actionKey(state, edge.action) == edge.semanticKey) return edge.action;
    } else {
        for (const auto action : Action::getAllActionsInState(state))
            if (action.isValidAction(state) && actionKey(state, action) == edge.semanticKey) return action;
    }
    throw std::runtime_error("shared public edge has no legal semantic action");
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
    if (child) child->evaluationPending = true;
}

void PublicBeliefCombatSearch::simulate(int particle, bool request, int rolloutTurns, int rolloutSteps) {
    BattleContext current = particles.at(particle);
    // Root particles carry public knowledge/latent state, not eight reusable future
    // worlds. Independent unknown futures avoid mistaking repeated determinization
    // for deep-search confidence. Preserve the single-particle deterministic API.
    if (particles.size() > 1) {
        const auto seed = (static_cast<std::uint64_t>(random()) << 32) | random();
        current.aiRng = Random(seed + 1); current.cardRandomRng = Random(seed + 2);
        current.miscRng = Random(seed + 3); current.monsterHpRng = Random(seed + 4);
        current.potionRng = Random(seed + 5); current.shuffleRng = Random(seed + 6);
        if (!current.player.hasRelic<RelicId::FROZEN_EYE>()) {
            auto &cards = current.cards.drawPile;
            auto first = cards.begin() + current.cards.knownDrawBottom;
            auto last = cards.end() - current.cards.knownDrawTop;
            if (current.inputState == InputState::CARD_SELECT
                && current.cardSelectInfo.cardSelectTask == CardSelectTask::SCRY)
                first = cards.begin() + std::min(std::max(current.cards.knownDrawBottom,
                    current.cardSelectInfo.pickCount), static_cast<int>(cards.size()) - current.cards.knownDrawTop);
            std::default_random_engine world(seed);
            std::shuffle(first, last, world);
        }
    }
    auto *at = nodes.at(rootKey).get();
    Path path;
    if (policyPriors && !rootRequested) {
        rootRequested = true;
        addRequest(current, std::move(path), at);
        return;
    }
    Node *leaf = nullptr;
    double fpuBase = 0;
    for (int depth = 0; depth < maximumActions && current.outcome == Outcome::UNDECIDED; ++depth) {
        if (std::any_of(path.begin(), path.end(), [&](const auto &p) { return p.first == at; }))
            throw std::runtime_error("cycle in public-search DAG");
        expand(*at, current);
        const auto choice = policyPriors ? selectPuct(*at, fpuBase) : select(*at);
        auto &edge = at->edges[choice];
        const bool unvisited = edge.visits + edge.inFlight == 0;
        const auto action = edgeAction(edge, current);
        ++at->inFlight; ++edge.inFlight;
        path.emplace_back(at, choice);
        action.execute(current);
        if (current.unsupportedEffectKind != UnsupportedEffectKind::NONE)
            throw std::runtime_error("unsupported effect in public-search transition");
        maximumTreeTurns = std::max(maximumTreeTurns, current.turn - particles.front().turn);
        if (current.outcome != Outcome::UNDECIDED) break;
        auto history = at->key;
        if (!isSettledSearchState(current)) {
            append(history, edge.semanticKey); append(history, observationKey(current));
        }
        const auto [key, existing] = findNode(current, history);
        if (edge.lastChildKey != key) {
            edge.children.insert(key);
            edge.lastChildKey = key;
        }
        // Keep the transition identity even when the rollout leaf is not stored:
        // another route may search that state later, making its evidence reachable
        // on rebase. Do not allocate a zero-evidence node just to remember the link.
        if (unvisited && !policyPriors) {
            boundedRollout(current, request ? rolloutTurns : -1, request ? rolloutSteps : maximumActions);
            break;
        }
        at = existing ? existing : &createNode(key);
        if (policyPriors && current.outcome == Outcome::UNDECIDED && !at->evaluated) {
            if (at->evaluationPending) {
                // Another path already requested this same public leaf. Avoid duplicate
                // inference; release virtual loss and let the rest of the batch proceed.
                for (const auto &[parent, index] : path) {
                    --parent->inFlight; --parent->edges[index].inFlight;
                }
                return;
            }
            expand(*at, current);
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
    treeActions += path.size();
    maximumTreeDepth = std::max(maximumTreeDepth, static_cast<int>(path.size()));
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
    return edgeAction(*best, particles.front());
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
    if (child) child->evaluationPending = false;
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
    if (found->second.child) found->second.child->evaluationPending = false;
    if (auto *child = found->second.child; child && !child->evaluated) {  // keeps uniform priors
        child->evaluated = true; child->value = value;
        if (child == nodes.at(rootKey).get()) applyRootNoise();
    }
    backup(found->second.path, value, &state);
    spareRequests.push_back(pending.extract(found));
}

void PublicBeliefCombatSearch::temperEvidence(Node &at, bool informative, std::int64_t visitLimit) {
    // Evidence is a warm start, not a promise about an unchanged hidden posterior.
    // Bound inertia by the previous root's actual compute; new information halves it.
    const double factor = std::min(1.0, static_cast<double>(visitLimit) / std::max<std::int64_t>(1, at.visits))
                          * (informative ? 0.5 : 1.0);
    at.visits = 0;
    for (auto &edge : at.edges) {
        const auto oldVisits = edge.visits;
        edge.visits = static_cast<std::int64_t>(oldVisits * factor);
        const double weight = oldVisits ? static_cast<double>(edge.visits) / oldVisits : 0.;
        edge.valueSum *= weight;
        const auto oldMeasured = edge.measured;
        edge.measured = static_cast<std::int64_t>(oldMeasured * weight);
        edge.wins = std::min(edge.measured, static_cast<std::int64_t>(edge.wins * weight));
        const double outcomeWeight = oldMeasured ? static_cast<double>(edge.measured) / oldMeasured : 0.;
        for (auto &value : edge.outcomeSum) value *= outcomeWeight;
        if (!edge.visits) edge.best = -std::numeric_limits<double>::infinity();
        at.visits += edge.visits;
    }
}

std::vector<BattleContext> PublicBeliefCombatSearch::advanceParticles(
    std::uint64_t semantic, const BattleContext &observed) const {
    const auto &edges = root().edges;
    const auto chosen = std::find_if(edges.begin(), edges.end(), [&](const Edge &e) {return e.semanticKey == semantic;});
    if (chosen == edges.end()) throw std::invalid_argument("committed action is not in the public root");
    const auto observation = publicSearchWords(observed);
    std::vector<BattleContext> survivors;
    for (auto state : particles) {
        edgeAction(*chosen, state).execute(state);
        if (state.unsupportedEffectKind != UnsupportedEffectKind::NONE)
            throw std::runtime_error("unsupported effect while advancing public particles");
        if (publicSearchWords(state) == observation) survivors.push_back(std::move(state));
    }
    return survivors;
}

void PublicBeliefCombatSearch::rebase(std::vector<BattleContext> states, std::uint64_t action, std::uint64_t seed) {
    if (states.empty() || !pending.empty()) throw std::invalid_argument("cannot rebase empty/pending search");
    if (std::none_of(root().edges.begin(), root().edges.end(), [&](const Edge &e) {return e.semanticKey == action;}))
        throw std::invalid_argument("rebase action was not legal at previous public root");
    const bool informative = states.front().turn != particles.front().turn
                             || states.front().cardsDrawn != particles.front().cardsDrawn;
    auto history = rootKey;
    append(history, action); append(history, observationKey(states.front()));
    particles = std::move(states);
    const auto publicState = publicSearchWords(particles.front());
    for (const auto &state : particles)
        if (publicSearchWords(state) != publicState)
            throw std::invalid_argument("rebased particles disagree on the public observation");
    random.seed(seed); rollout.randGen.seed(seed);
    rootKey = node(particles.front(), history).key;
    expand(*nodes.at(rootKey), particles.front());
    // Preserve up to one previous root's compute, with an ordinary-budget floor.
    // A fixed 256 cap would erase almost all confidence from deep boss searches.
    if (simulations > 0) retentionBudget = std::max<std::int64_t>(256, simulations);
    const auto visitLimit = retentionBudget;
    std::unordered_set<std::uint64_t> keep;
    std::vector<std::uint64_t> todo{rootKey};
    while (!todo.empty()) {
        const auto key = todo.back(); todo.pop_back();
        if (!keep.insert(key).second) continue;
        const auto found = nodes.find(key);
        if (found == nodes.end()) continue; // identified but never searched rollout leaf
        for (const auto &edge : found->second->edges)
            for (const auto child : edge.children) todo.push_back(child);
    }
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (!keep.count(it->first)) it = nodes.erase(it);
        else { temperEvidence(*it->second, informative, visitLimit); ++it; }
    }
    retainedNodes = nodes.size(); retainedVisits = root().visits;
    if (policyPriors) {
        rootRequested = root().evaluated; rootNoised = false;
        noiseRandom.seed(seed ^ 0x6e6f697365ULL);
        applyRootNoise();
    }
    simulations = terminalEvaluations = unresolvedEvaluations = neuralEvaluations = treeActions = 0;
    maximumTreeDepth = maximumTreeTurns = 0;
}
}
