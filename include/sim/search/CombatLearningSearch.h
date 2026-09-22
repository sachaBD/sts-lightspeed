#ifndef STS_COMBAT_LEARNING_SEARCH_H
#define STS_COMBAT_LEARNING_SEARCH_H

#include "sim/search/PublicBeliefCombatSearch.h"
#include <array>
#include <map>

namespace sts::search {

// Undiscounted, separately observable fight outcomes. No damage/block shaping.
// win, surviving HP/100, surviving potion count/5, max HP/100, gold/1000.
using CombatOutcome = std::array<double, 5>;

struct CombatObjective {
    double survival = 10.0;
    double potionHp = 6.0;
    double maxHpPrice = 0.5;
    double goldHpPrice = 0.02;
    double score(const CombatOutcome &value) const;
};

class CombatLearningSearch {
public:
    struct Edge {
        Action action;
        std::uint64_t semantic = 0;
        double prior = 0;
        std::int64_t visits = 0;
        int inFlight = 0;
        CombatOutcome sum{};
    };
    struct Node {
        bool expanded = false;
        bool pending = false;
        std::int64_t visits = 0;
        int inFlight = 0;
        std::vector<Edge> edges;
    };
    using Path = std::vector<std::pair<Node *, std::size_t>>;
    struct Request {
        std::uint64_t id;
        Node *node;
        BattleContext state;
        Path path;
    };

    std::vector<BattleContext> particles;
    BattleContext observed;
    CombatObjective objective;
    std::map<std::uint64_t, Request> pending;
    std::int64_t simulations = 0;
    std::int64_t terminalEvaluations = 0;
    std::int64_t unresolvedEvaluations = 0;
    int maximumActions = 512;

    CombatLearningSearch(std::vector<BattleContext> states, const BattleContext &source,
                         std::uint64_t seed, CombatObjective target);
    std::vector<std::uint64_t> requestBatch(int count, std::int64_t totalBudget);
    void submit(std::uint64_t id, const std::vector<double> &priors, const CombatOutcome &value);
    void submitGuided(std::uint64_t id, const std::vector<double> &priors);
    void searchGuided(std::int64_t totalBudget);
    const Node &root() const;
    std::vector<Action> requestActions(const Request &request) const;
    Action selectedAction() const;
    static CombatOutcome terminalOutcome(const BattleContext &state);
private:
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes;
    std::default_random_engine random;
    BattleScumSearcher2 rollout;
    std::uint64_t nextRequest = 0;
    Node &node(std::uint64_t key, const BattleContext &state);
    std::size_t select(const Node &node) const;
    bool requestOne();
    void backup(Path &path, const CombatOutcome &value);
    static void release(Path &path);
};

}
#endif
