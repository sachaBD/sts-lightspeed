#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#ifdef PBCS_PROFILE
#include <x86intrin.h>
#define PROF_START(t) auto t = __rdtsc()
#define PROF_ADD(i, t) g_prof[i] += __rdtsc() - t
#else
#define PROF_START(t)
#define PROF_ADD(i, t)
#endif
namespace sts::search {
unsigned long long g_prof[8];  // PBCS_PROFILE: cycles in simulate's phases
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
template<class Pile> void pile(std::uint64_t &hash, const Pile &cards, bool unordered) {
    std::vector<std::uint64_t> keys;
    for (const auto &card : cards) keys.push_back(cardKey(card));
    if (unordered) std::sort(keys.begin(), keys.end());
    append(hash, keys.size());
    for (auto key : keys) append(hash, key);
}
// Fast observation key (node keys only): the same fields and the same equality as publicObservation,
// with one 64-bit mix per value instead of byte-wise FNV.
std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
// Position-tagged sum of mixed words: the adds are independent (no serial dependency chain), so they
// overlap in the pipeline. Two different word sequences give the same sum only by 64-bit coincidence.
std::uint64_t positionTag(std::size_t i) { return mix64((i + 1) * 0x9E3779B97F4A7C15ULL); }
struct PositionTags {
    std::array<std::uint64_t, 1024> tags{};
    PositionTags() { for (std::size_t i = 0; i < tags.size(); ++i) tags[i] = positionTag(i); }
};
const PositionTags POSITION_TAGS;
struct FastHash {
    std::uint64_t sum = 0;
    std::size_t position = 0;
    void add(std::uint64_t value) {
        const auto tag = position < POSITION_TAGS.tags.size() ? POSITION_TAGS.tags[position] : positionTag(position);
        ++position;
        sum += mix64(value ^ tag);
    }
    std::uint64_t value() const { return mix64(sum ^ position); }
};
// Injective packing of cardKey's fields (id, upgrade count, specialData, cost, costForTurn,
// freeToPlayOnce, retain) into one word.
std::uint64_t cardWord(const CardInstance &card) {
    return static_cast<std::uint64_t>(static_cast<std::uint16_t>(card.id))
        | static_cast<std::uint64_t>(static_cast<std::uint16_t>(card.specialData)) << 16
        | static_cast<std::uint64_t>(static_cast<std::uint8_t>(card.cost)) << 32
        | static_cast<std::uint64_t>(static_cast<std::uint8_t>(card.costForTurn)) << 40
        | static_cast<std::uint64_t>(card.getUpgradeCount() & 0x3fff) << 48
        | static_cast<std::uint64_t>(card.freeToPlayOnce) << 62
        | static_cast<std::uint64_t>(card.retain) << 63;
}
template<class Pile> void fastPile(FastHash &hash, const Pile &cards, bool unordered) {
    hash.add(cards.size());
    if (unordered) {  // multiset: order-free sum of mixed card words
        std::uint64_t sum = 0;
        for (const auto &card : cards) sum += mix64(cardWord(card) ^ 0x5851f42d4c957f2dULL);
        hash.add(sum);
    } else {
        for (const auto &card : cards) hash.add(cardWord(card));
    }
}
bool drawSelection(const BattleContext &state) {
    if (state.inputState != InputState::CARD_SELECT) return false;
    switch (state.cardSelectInfo.cardSelectTask) {
        case CardSelectTask::SECRET_TECHNIQUE: case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK: case CardSelectTask::OMNISCIENCE: return true;
        default: return false;
    }
}
}

std::uint64_t PublicBeliefCombatSearch::publicActionKey(const BattleContext &state, Action action) {
    if (drawSelection(state) && action.getActionType() == ActionType::SINGLE_CARD_SELECT) {
        auto hash = ROOT_KEY;
        append(hash, static_cast<int>(action.getActionType()));
        append(hash, cardKey(state.cards.drawPile.at(action.getSelectIdx())));
        return hash;
    }
    return action.bits;
}

std::uint64_t PublicBeliefCombatSearch::actionKey(const BattleContext &state, Action action) const {
    return mergeIdenticalCards ? identityActionKey(state, action) : publicActionKey(state, action);
}

std::uint64_t PublicBeliefCombatSearch::identityActionKey(const BattleContext &state, Action action) {
    if (state.inputState == InputState::PLAYER_NORMAL) {
        const auto type = action.getActionType();
        if (type == ActionType::CARD || type == ActionType::POTION) {
            auto hash = ROOT_KEY;
            append(hash, 0x6d65726765ULL);  // domain tag: never equal to raw action bits
            append(hash, static_cast<int>(type));
            if (type == ActionType::CARD) append(hash, cardKey(state.cards.hand[action.getSourceIdx()]));
            else append(hash, static_cast<int>(state.potions[action.getSourceIdx()]));
            append(hash, action.getTargetIdx());
            return hash;
        }
    }
    return publicActionKey(state, action);
}

Action PublicBeliefCombatSearch::mapAction(const BattleContext &source, Action action,
                                         const BattleContext &target) {
    if (!drawSelection(source)) return action;
    const auto key = publicActionKey(source, action);
    for (const auto &candidate : Action::enumerateCardSelectActions(target)) {
        if (candidate.isValidAction(target) && publicActionKey(target, candidate) == key) return candidate;
    }
    throw std::runtime_error("public draw-selection action has no semantic match");
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

std::uint64_t PublicBeliefCombatSearch::publicObservation(const BattleContext &s) {
    std::uint64_t hash = ROOT_KEY;
    // This is an observation appended to the action/observation history,
    // not a claim that this reduced vector is a complete Markov state.
    // Seed, all RNGs, draw order, internal enemy miscInfo and debug counters
    // are deliberately excluded. Known deterministic effects are already
    // represented by the common preceding history and the chosen action.
    append(hash, s.turn); append(hash, static_cast<int>(s.inputState));
    append(hash, static_cast<int>(s.outcome));
    const auto &p = s.player;
    append(hash, p.curHp); append(hash, p.maxHp); append(hash, p.energy);
    append(hash, p.energyPerTurn); append(hash, p.block); append(hash, p.strength);
    append(hash, p.dexterity); append(hash, p.focus); append(hash, p.artifact);
    append(hash, p.statusBits0); append(hash, p.statusBits1);
    for (const auto &entry : p.statusMap) {
        append(hash, static_cast<int>(entry.first)); append(hash, entry.second);
    }
    append(hash, p.cardsPlayedThisTurn); append(hash, p.attacksPlayedThisTurn);
    append(hash, p.skillsPlayedThisTurn); append(hash, p.cardsDiscardedThisTurn);
    append(hash, p.happyFlowerCounter); append(hash, p.incenseBurnerCounter);
    append(hash, p.inkBottleCounter); append(hash, p.nunchakuCounter);
    append(hash, p.penNibCounter); append(hash, p.sundialCounter);
    append(hash, p.relicBits0); append(hash, p.relicBits1);
    append(hash, s.potionCapacity);
    for (int i = 0; i < s.potionCapacity; ++i) append(hash, static_cast<int>(s.potions[i]));
    append(hash, s.cards.cardsInHand);
    for (int i = 0; i < s.cards.cardsInHand; ++i) append(hash, cardKey(s.cards.hand[i]));
    pile(hash, s.cards.drawPile, !p.hasRelic<RelicId::FROZEN_EYE>());
    pile(hash, s.cards.discardPile, false); pile(hash, s.cards.exhaustPile, false);
    append(hash, s.monsters.monsterCount);
    for (int i = 0; i < s.monsters.monsterCount; ++i) {
        const auto &m = s.monsters.arr[i];
        append(hash, static_cast<int>(m.id)); append(hash, m.curHp);
        append(hash, m.maxHp); append(hash, m.block);
        append(hash, m.isTargetable()); append(hash, m.halfDead);
        append(hash, m.statusBits); append(hash, m.artifact);
        append(hash, m.strength); append(hash, m.weak); append(hash, m.vulnerable);
        append(hash, m.metallicize); append(hash, m.platedArmor); append(hash, m.poison);
        append(hash, m.regen); append(hash, m.shackled);
        if (!p.hasRelic<RelicId::RUNIC_DOME>()) {
            append(hash, static_cast<int>(m.moveHistory[0]));
            if (m.isTargetable()) {
                const auto intent = m.getMoveBaseDamage(s);
                append(hash, intent.damage); append(hash, intent.attackCount);
            }
        }
        append(hash, static_cast<int>(m.moveHistory[1]));
    }
    if (s.inputState == InputState::CARD_SELECT) {
        const auto &selection = s.cardSelectInfo;
        append(hash, static_cast<int>(selection.cardSelectTask));
        append(hash, selection.pickCount); append(hash, selection.canPickZero);
        append(hash, selection.canPickAnyNumber);
        append(hash, selection.data0);
        if (selection.cardSelectTask == CardSelectTask::SCRY) {
            for (int i = 0; i < std::min(selection.pickCount, static_cast<int>(s.cards.drawPile.size())); ++i)
                append(hash, cardKey(s.cards.drawPile[i]));
        }
        if (selection.cardSelectTask == CardSelectTask::DISCOVERY
            || selection.cardSelectTask == CardSelectTask::CODEX) {
            for (const auto id : selection.cards) append(hash, static_cast<int>(id));
        }
    }
    return hash;
}

std::uint64_t PublicBeliefCombatSearch::observationKey(const BattleContext &s) {
    // Mirrors publicObservation field for field (keep them in sync).
    FastHash hash;
    hash.add(s.turn); hash.add(static_cast<int>(s.inputState));
    hash.add(static_cast<int>(s.outcome));
    const auto &p = s.player;
    hash.add(p.curHp); hash.add(p.maxHp); hash.add(p.energy);
    hash.add(p.energyPerTurn); hash.add(p.block); hash.add(p.strength);
    hash.add(p.dexterity); hash.add(p.focus); hash.add(p.artifact);
    hash.add(p.statusBits0); hash.add(p.statusBits1);
    hash.add(p.statusMap.size());
    for (const auto &entry : p.statusMap) {
        hash.add(static_cast<int>(entry.first)); hash.add(entry.second);
    }
    hash.add(p.cardsPlayedThisTurn); hash.add(p.attacksPlayedThisTurn);
    hash.add(p.skillsPlayedThisTurn); hash.add(p.cardsDiscardedThisTurn);
    hash.add(p.happyFlowerCounter); hash.add(p.incenseBurnerCounter);
    hash.add(p.inkBottleCounter); hash.add(p.nunchakuCounter);
    hash.add(p.penNibCounter); hash.add(p.sundialCounter);
    hash.add(p.relicBits0); hash.add(p.relicBits1);
    hash.add(s.potionCapacity);
    for (int i = 0; i < s.potionCapacity; ++i) hash.add(static_cast<int>(s.potions[i]));
    hash.add(s.cards.cardsInHand);
    for (int i = 0; i < s.cards.cardsInHand; ++i) hash.add(cardWord(s.cards.hand[i]));
    fastPile(hash, s.cards.drawPile, !p.hasRelic<RelicId::FROZEN_EYE>());
    fastPile(hash, s.cards.discardPile, false); fastPile(hash, s.cards.exhaustPile, false);
    hash.add(s.monsters.monsterCount);
    for (int i = 0; i < s.monsters.monsterCount; ++i) {
        const auto &m = s.monsters.arr[i];
        hash.add(static_cast<int>(m.id)); hash.add(m.curHp);
        hash.add(m.maxHp); hash.add(m.block);
        hash.add(m.isTargetable()); hash.add(m.halfDead);
        hash.add(m.statusBits); hash.add(m.artifact);
        hash.add(m.strength); hash.add(m.weak); hash.add(m.vulnerable);
        hash.add(m.metallicize); hash.add(m.platedArmor); hash.add(m.poison);
        hash.add(m.regen); hash.add(m.shackled);
        if (!p.hasRelic<RelicId::RUNIC_DOME>()) {
            hash.add(static_cast<int>(m.moveHistory[0]));
            if (m.isTargetable()) {
                const auto intent = m.getMoveBaseDamage(s);
                hash.add(intent.damage); hash.add(intent.attackCount);
            }
        }
        hash.add(static_cast<int>(m.moveHistory[1]));
    }
    if (s.inputState == InputState::CARD_SELECT) {
        const auto &selection = s.cardSelectInfo;
        hash.add(static_cast<int>(selection.cardSelectTask));
        hash.add(selection.pickCount); hash.add(selection.canPickZero);
        hash.add(selection.canPickAnyNumber);
        hash.add(selection.data0);
        if (selection.cardSelectTask == CardSelectTask::SCRY) {
            for (int i = 0; i < std::min(selection.pickCount, static_cast<int>(s.cards.drawPile.size())); ++i)
                hash.add(cardWord(s.cards.drawPile[i]));
        }
        if (selection.cardSelectTask == CardSelectTask::DISCOVERY
            || selection.cardSelectTask == CardSelectTask::CODEX) {
            for (const auto id : selection.cards) hash.add(static_cast<int>(id));
        }
    }
    return hash.value();
}

PublicBeliefCombatSearch::PublicBeliefCombatSearch(
    std::vector<BattleContext> states, std::uint64_t seed, int rolloutMode, bool mergeIdentical)
    : particles(std::move(states)), random(seed), rollout(particles.at(0)),
      normalization(100.0 * (35 + particles.at(0).player.maxHp + 20)) {
    mergeIdenticalCards = mergeIdentical;
    rollout.rolloutMode = rolloutMode;
    rollout.randGen.seed(seed);
    node(ROOT_KEY, particles.front());
}

PublicBeliefCombatSearch::Node &PublicBeliefCombatSearch::node(
    std::uint64_t key, const BattleContext &state) {
    auto found = nodes.find(key);
    if (found != nodes.end()) return *found->second;
    auto created = std::make_unique<Node>();
    BattleScumSearcher2::Node temporary;
    rollout.enumerateActionsForNode(temporary, state);
    for (const auto &edge : temporary.edges) {
        if (!edge.action.isValidAction(state)) {
            throw std::runtime_error("native enumeration produced an illegal belief-search action");
        }
        const auto semanticKey = actionKey(state, edge.action);
        if (std::none_of(created->edges.begin(), created->edges.end(),
                        [&](const auto &item) { return item.semanticKey == semanticKey; })) {
            created->edges.push_back({edge.action, semanticKey});
        }
    }
    if (created->edges.empty()) throw std::runtime_error("empty belief-search node");
    if (drawSelection(state)) {
        std::sort(created->edges.begin(), created->edges.end(),
                  [](const auto &a, const auto &b) { return a.semanticKey < b.semanticKey; });
    }
    std::shuffle(created->edges.begin(), created->edges.end(), random);
    auto *result = created.get(); nodes.emplace(key, std::move(created));
    return *result;
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

void PublicBeliefCombatSearch::simulate(int particle, bool request, int rolloutTurns, int rolloutSteps) {
    PROF_START(T0);
    BattleContext current(particles[particle]);
    PROF_ADD(0, T0);
    std::uint64_t key = rootKey;
    Path path;
    for (int depth = 0; depth < maximumActions; ++depth) {
        if (current.outcome != Outcome::UNDECIDED) break;
        PROF_START(T1);
        auto &at = node(key, current);
        PROF_ADD(1, T1);
        PROF_START(T1b);
        auto choice = select(at);
        PROF_ADD(2, T1b);
        auto action = at.edges[choice].action;
        if (drawSelection(current)) {
            bool matched = false;
            for (const auto &candidate : Action::enumerateCardSelectActions(current)) {
                if (candidate.isValidAction(current)
                    && publicActionKey(current, candidate) == at.edges[choice].semanticKey) {
                    action = candidate; matched = true; break;
                }
            }
            if (!matched) throw std::runtime_error("shared belief edge has no public semantic action");
        }
        if (!action.isValidAction(current)) {
            std::ostringstream message;
            message << "public observation aliases incompatible legal actions: floor="
                << current.floorNum << " depth=" << depth << " visits=" << at.visits
                << " input=" << static_cast<int>(current.inputState)
                << " task=" << static_cast<int>(current.cardSelectInfo.cardSelectTask)
                << " action=" << action.bits << " legal=";
            for (const auto &legal : Action::getAllActionsInState(current)) message << legal.bits << ',';
            throw std::runtime_error(message.str());
        }
        const bool unvisited = at.edges[choice].visits + at.edges[choice].inFlight == 0;
        ++at.inFlight; ++at.edges[choice].inFlight;
        path.emplace_back(&at, choice);
        PROF_START(T2);
        action.execute(current);
        PROF_ADD(3, T2);
        if (unvisited) {
            boundedRollout(current, request ? rolloutTurns : -1, request ? rolloutSteps : maximumActions);
            break;
        }
        PROF_START(T3);
        append(key, at.edges[choice].semanticKey); append(key, observationKey(current));
        PROF_ADD(4, T3);
        at.children.insert(key);
    }
    if (current.unsupportedEffectKind != UnsupportedEffectKind::NONE)
        throw std::runtime_error("unsupported effect in unified public combat search");
    if (request && current.outcome == Outcome::UNDECIDED) {
        PROF_START(T4);
        // Reuse a submitted request's map node: no allocation, and copy-assignment keeps the state's
        // pile capacity.
        if (spareRequests.empty()) {
            auto &slot = pending.emplace_hint(pending.end(), std::piecewise_construct,
                                              std::forward_as_tuple(++nextRequest), std::forward_as_tuple())->second;
            slot.state = current; slot.path = std::move(path);
        } else {
            auto handle = std::move(spareRequests.back());
            spareRequests.pop_back();
            handle.key() = ++nextRequest;
            handle.mapped().state = current; handle.mapped().path = std::move(path);
            pending.insert(pending.end(), std::move(handle));
        }
        PROF_ADD(5, T4);
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
        temporary.edges.clear();
        rollout.enumerateActionsForNode(temporary, state);
        if (temporary.edges.empty()) throw std::runtime_error("empty public rollout support");
        const auto choice = rollout.selectRolloutAction(temporary, state);
        temporary.edges[choice].action.execute(state);
    }
    // An unfinished rollout is unresolved evidence, never a victory.
}

void PublicBeliefCombatSearch::search(std::int64_t budget) {
    if (budget <= 0 || !pending.empty()) throw std::invalid_argument("invalid or pending public search budget");
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
    auto &edges = nodes.at(rootKey)->edges;
    if (priors.size() != edges.size() || !std::isfinite(strength) || strength < 0)
        throw std::invalid_argument("invalid unified root prior");
    double sum = 0;
    for (double p : priors) {
        if (!std::isfinite(p) || p < 0) throw std::invalid_argument("invalid prior probability");
        sum += p;
    }
    if (!(sum > 0)) throw std::invalid_argument("empty prior support");
    for (std::size_t i = 0; i < edges.size(); ++i) edges[i].prior = priors[i] / sum;
    priorStrength = strength;
}

void PublicBeliefCombatSearch::setObjective(int mode, double victory, double potion, double maximumHp, double gold) {
    if (simulations || root().visits || !pending.empty() || mode < 0 || mode > 1)
        throw std::invalid_argument("objective is immutable once evidence exists");
    for (double v : {victory, potion, maximumHp, gold})
        if (!std::isfinite(v) || v < 0) throw std::invalid_argument("invalid objective coefficient");
    objectiveMode = mode; victoryHp = victory; potionHp = potion; maxHpPrice = maximumHp; goldHpPrice = gold;
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
    const auto first = nextRequest;
    std::uniform_int_distribution<int> sample(0, static_cast<int>(particles.size())-1);
    for (int n = 0; n < count && simulations + static_cast<std::int64_t>(pending.size()) < budget; ++n)
        simulate(particles.size() == 1 ? 0 : sample(random), true, turns, steps);
    std::vector<std::uint64_t> result;
    for (const auto &[id, request] : pending) if (id > first) result.push_back(id);
    return result;
}

void PublicBeliefCombatSearch::submit(std::uint64_t id, double value) {
    auto found = pending.find(id);
    if (found == pending.end() || !std::isfinite(value) || value < -.1 || value > 2)
        throw std::invalid_argument("invalid unified leaf submission");
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
    backup(found->second.path, terminalValue(state), &state);
    spareRequests.push_back(pending.extract(found));
}

void PublicBeliefCombatSearch::rebase(std::vector<BattleContext> states, std::uint64_t action, std::uint64_t seed) {
    if (states.empty() || !pending.empty()) throw std::invalid_argument("cannot rebase empty/pending search");
    if (std::none_of(root().edges.begin(), root().edges.end(), [&](const Edge &e) {return e.semanticKey == action;}))
        throw std::invalid_argument("rebase action was not legal at previous public root");
    append(rootKey, action); append(rootKey, observationKey(states.front()));
    particles = std::move(states);
    random.seed(seed); rollout.randGen.seed(seed);
    node(rootKey, particles.front());
    std::unordered_set<std::uint64_t> keep;
    std::vector<std::uint64_t> todo{rootKey};
    while (!todo.empty()) {
        auto key = todo.back(); todo.pop_back();
        auto found = nodes.find(key);
        if (found == nodes.end() || !keep.insert(key).second) continue;
        for (auto child : found->second->children) todo.push_back(child);
    }
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (!keep.count(it->first)) it = nodes.erase(it); else ++it;
    }
    retainedNodes = nodes.size(); retainedVisits = root().visits;
    simulations = terminalEvaluations = unresolvedEvaluations = neuralEvaluations = 0;
}
}
