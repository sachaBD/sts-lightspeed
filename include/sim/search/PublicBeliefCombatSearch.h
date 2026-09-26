#ifndef STS_PUBLIC_BELIEF_COMBAT_SEARCH_H
#define STS_PUBLIC_BELIEF_COMBAT_SEARCH_H

#include "sim/search/BattleScumSearcher2.h"
#include <cstdint>
#include <array>
#include <limits>
#include <map>
#include <unordered_set>
#include <unordered_map>

namespace sts::search {

// Experimental root-sampled action/observation-history tree. All particles
// share one policy at identical public histories, unlike independent perfect-
// information searches followed by an average of their root values.
struct PublicBeliefCombatSearch {
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
    };
    struct Node {
        std::int64_t visits = 0;
        std::vector<Edge> edges;
        int inFlight = 0;
        std::unordered_set<std::uint64_t> children;
    };
    using Path = std::vector<std::pair<Node *, std::size_t>>;
    struct Request { BattleContext state; Path path; };
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
    double priorStrength = 0.0;
    // Max backup (single-particle / deterministic search): selection and selectedAction use each
    // edge's best backed-up value instead of its mean. Off = the historical mean backup.
    bool maxBackup = false;
    // (Constructor argument: the root's edges are built there.) Merge edges that play identical cards (same cardKey) from different hand slots, or drink
    // identical potions from different slots, at the same target: one edge instead of several that
    // split the visits of one move. Off = the historical per-slot edges.
    bool mergeIdenticalCards = false;
    // Zero is the unmodified historical control. One has explicit terminal
    // resources and excludes escapes; scoring changes are independent of NN use.
    int objectiveMode = 0;
    double victoryHp = 35.0, potionHp = 4.0, maxHpPrice = 0.0, goldHpPrice = 0.0;
    std::map<std::uint64_t, Request> pending;
    std::vector<std::map<std::uint64_t, Request>::node_type> spareRequests;  // submitted, for reuse

    PublicBeliefCombatSearch(std::vector<BattleContext> states,
                            std::uint64_t seed, int rolloutMode, bool mergeIdenticalCards = false);
    void search(std::int64_t budget);
    const Node &root() const;
    Action selectedAction() const;
    void setRootPrior(const std::vector<double> &priors, double strength);
    void setObjective(int mode, double victory, double potion, double maximumHp, double gold);
    std::vector<std::uint64_t> requestBatch(int count, std::int64_t totalBudget,
                                          int rolloutTurns, int rolloutSteps);
    void submit(std::uint64_t request, double normalizedValue);
    void submitGuided(std::uint64_t request);
    double scorePrediction(double win, double hp, double potions, double maximumHp, double gold) const;
    void rebase(std::vector<BattleContext> states, std::uint64_t semanticAction, std::uint64_t seed);
    static std::uint64_t publicObservation(const BattleContext &state);
    // Tree node keys: publicObservation's equality (same fields) with a faster hash. Seeds still use
    // publicObservation, so search results are unchanged.
    static std::uint64_t observationKey(const BattleContext &state);
    static std::uint64_t publicActionKey(const BattleContext &state, Action action);
    // This search's edge key for `action` at `state` (publicActionKey, merged per mergeIdenticalCards).
    std::uint64_t actionKey(const BattleContext &state, Action action) const;
    // publicActionKey, except card plays / potions keyed by card (cardKey) / potion and target, not slot.
    static std::uint64_t identityActionKey(const BattleContext &state, Action action);
    static Action mapAction(const BattleContext &source, Action action, const BattleContext &target);
    static void resampleDraw(BattleContext &target, const BattleContext &observed, std::uint64_t seed);
private:
    std::uint64_t rootKey = 0xcbf29ce484222325ULL, nextRequest = 0;
    void simulate(int particle, bool request = false, int rolloutTurns = -1, int rolloutSteps = 512);
    Node &node(std::uint64_t key, const BattleContext &state);
    std::size_t select(const Node &node);
    void boundedRollout(BattleContext &state, int turns = -1, int steps = 512);
    void backup(Path &path, double value, const BattleContext *terminal);
    double terminalValue(const BattleContext &state) const;
};

}
#endif
