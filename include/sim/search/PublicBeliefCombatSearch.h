#ifndef STS_PUBLIC_BELIEF_COMBAT_SEARCH_H
#define STS_PUBLIC_BELIEF_COMBAT_SEARCH_H

#include "sim/search/BattleScumSearcher2.h"
#include "sim/search/SearchState.h"
#include <cstdint>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <unordered_set>
#include <unordered_map>

namespace sts::search {

// Persistent public-state search DAG. Equivalent public decision problems share
// continuation evidence; each simulation still executes its own sampled hidden future.
// This deliberately approximates belief-history equality, not simulator transitions.
struct PublicBeliefCombatSearch {
    struct Node;
    struct Edge {
        Action action;
        std::uint64_t semanticKey = 0;
        std::int64_t visits = 0;
        double valueSum = 0;
        std::int64_t wins = 0;
        double prior = 0.0;
        int inFlight = 0;
        std::int64_t measured = 0;
        double best = -std::numeric_limits<double>::infinity();  // max backed-up value
        // Actual terminal win, surviving HP, potions, max HP, gold; not guesses.
        std::array<double, 5> outcomeSum{};
        std::unordered_set<std::uint64_t> children;
        std::optional<std::uint64_t> lastChildKey; // membership cache, not a hidden transition
    };
    struct Node {
        std::uint64_t key = 0;
        std::vector<std::uint64_t> signature;
        bool expanded = false;
        std::int64_t visits = 0;
        std::vector<Edge> edges;
        int inFlight = 0;
        // Policy-prior mode only: the leaf evaluation of this node has been submitted, and its value.
        // Until then its edges carry uniform priors.
        bool evaluated = false, evaluationPending = false;
        double value = 0;
    };
    using Path = std::vector<std::pair<Node *, std::size_t>>;
    // `child` (policy-prior mode only): the node created for this leaf, whose edges (in order) the
    // priors of submit(id, value, priors) refer to. Null in the historical mode, and for a leaf cut
    // off by maximumActions.
    struct Request { BattleContext state; Path path; Node *child = nullptr; };
    std::vector<BattleContext> particles;
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes;
    std::default_random_engine random;
    BattleScumSearcher2 rollout;
    double normalization;
    int maximumActions = 512;
    double exploration = 0.7;
    std::int64_t simulations = 0;
    std::int64_t terminalEvaluations = 0, unresolvedEvaluations = 0, neuralEvaluations = 0;
    std::int64_t retainedVisits = 0;
    std::size_t retainedNodes = 0;
    std::int64_t transpositionHits = 0, treeActions = 0;
    int maximumTreeDepth = 0, maximumTreeTurns = 0;
    double priorStrength = 0.0;
    // Policy-prior mode (enablePolicyPriors): every node is expanded when its leaf is evaluated and
    // selects by PUCT with first-play urgency. Otherwise selects by UCB1.
    bool policyPriors = false;
    double cPuct = 1.5, fpuReduction = 0.15;
    double dirichletAlpha = 0.0, dirichletEpsilon = 0.0;  // root noise, off by default
    // Max backup (single-particle / deterministic search): selection and selectedAction use each
    // edge's best backed-up value instead of its mean.
    bool maxBackup = false;
    // Zero is the unmodified historical control. One has explicit terminal
    // resources and excludes escapes; scoring changes are independent of NN use.
    int objectiveMode = 0;
    double victoryHp = 35.0, potionHp = 4.0, maxHpPrice = 0.0, goldHpPrice = 0.0;
    std::map<std::uint64_t, Request> pending;
    std::vector<std::map<std::uint64_t, Request>::node_type> spareRequests;  // submitted, for reuse

    PublicBeliefCombatSearch(std::vector<BattleContext> states,
                            std::uint64_t seed, int rolloutMode);
    void search(std::int64_t budget);
    const Node &root() const;
    Action selectedAction() const;
    void setRootPrior(const std::vector<double> &priors, double strength);
    // Requires objectiveMode 1, no evidence yet, and no root prior. Only requestBatch(.., 0, 0) and
    // submit/submitGuided drive the search afterwards (search() throws). The root itself is the first
    // request. noiseSeed seeds the Dirichlet noise only (rebase reseeds it from its seed); the
    // search's own random stream is untouched.
    void enablePolicyPriors(double cPuct, double fpuReduction, double dirichletAlpha = 0.0,
                            double dirichletEpsilon = 0.0, std::uint64_t noiseSeed = 0);
    // Root max HP the value scale was fixed with (constructor / setObjective); unchanged by rebase.
    // objectiveMode 1 terminal value: won ? 100*(victoryHp + hp + potionHp*potions + ...) / normalization : 0,
    // normalization = 100*(victoryHp + M0 + 5*potionHp + maxHpPrice*M0 + goldHpPrice*gold0 + 1).
    int objectiveMaxHp() const { return normalizationMaxHp; }
    void setObjective(int mode, double victory, double potion, double maximumHp, double gold);
    std::vector<std::uint64_t> requestBatch(int count, std::int64_t totalBudget,
                                          int rolloutTurns, int rolloutSteps);
    void submit(std::uint64_t request, double normalizedValue);
    // Policy-prior mode: also sets the leaf node's priors (aligned to request.child->edges, finite,
    // >= 0, normalized here). First evaluation of a node wins; submit without priors keeps uniform.
    void submit(std::uint64_t request, double normalizedValue, const std::vector<double> &priors);
    void submitGuided(std::uint64_t request);
    double scorePrediction(double win, double hp, double potions, double maximumHp, double gold) const;
    // Warm-start evidence, never old hidden transitions. Root evidence is bounded and
    // discounted on new turns/draws so fresh simulations can correct stale estimates.
    void rebase(std::vector<BattleContext> states, std::uint64_t semanticAction, std::uint64_t seed);
    std::vector<BattleContext> advanceParticles(std::uint64_t semanticAction,
                                               const BattleContext &observed) const;
    Action edgeAction(const Edge &edge, const BattleContext &state) const;
    static std::uint64_t publicObservation(const BattleContext &state);
    static std::uint64_t observationKey(const BattleContext &state);
    static std::uint64_t publicActionKey(const BattleContext &state, Action action);
    std::uint64_t actionKey(const BattleContext &state, Action action) const;
    static std::uint64_t identityActionKey(const BattleContext &state, Action action);
    static Action mapAction(const BattleContext &source, Action action, const BattleContext &target);
    static void resampleDraw(BattleContext &target, const BattleContext &observed, std::uint64_t seed);
private:
    std::uint64_t rootKey = 0xcbf29ce484222325ULL, nextRequest = 0;
    int normalizationMaxHp = 0;
    bool rootRequested = false, rootNoised = false;  // policy-prior mode
    std::mt19937_64 noiseRandom;
    void simulate(int particle, bool request = false, int rolloutTurns = -1, int rolloutSteps = 512);
    void addRequest(const BattleContext &state, Path &&path, Node *child);
    std::vector<std::uint64_t> signatureScratch;
    std::int64_t retentionBudget = 256;
    void nodeWords(const BattleContext &state, std::uint64_t history);
    std::pair<std::uint64_t, Node *> findNode(const BattleContext &state, std::uint64_t history);
    Node &createNode(std::uint64_t key);
    Node &node(const BattleContext &state, std::uint64_t history);
    void expand(Node &node, const BattleContext &state);
    static void temperEvidence(Node &node, bool informative, std::int64_t visitLimit);
    std::size_t select(const Node &node);
    std::size_t selectPuct(const Node &node, double &fpuBase) const;
    void applyRootNoise();
    void boundedRollout(BattleContext &state, int turns = -1, int steps = 512);
    void backup(Path &path, double value, const BattleContext *terminal);
    double terminalValue(const BattleContext &state) const;
};

}
#endif
