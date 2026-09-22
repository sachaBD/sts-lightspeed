#include "sim/search/CombatLearningSearch.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sts::search {
namespace {
constexpr std::uint64_t ROOT = 0xcbf29ce484222325ULL;
void append(std::uint64_t &hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { hash ^= (value >> (8*i)) & 255; hash *= 0x100000001b3ULL; }
}
Action match(const BattleContext &state, const CombatLearningSearch::Edge &edge) {
    for (auto action : Action::getAllActionsInState(state)) {
        if (action.isValidAction(state)
            && PublicBeliefCombatSearch::publicActionKey(state, action) == edge.semantic) return action;
    }
    throw std::runtime_error("combat learning tree aliases incompatible public actions");
}
}

double CombatObjective::score(const CombatOutcome &v) const {
    return (survival*v[0] + v[1] + potionHp*0.05*v[2]
            + maxHpPrice*v[3] + goldHpPrice*10.0*v[4]) / (survival + 4.0);
}

CombatOutcome CombatLearningSearch::terminalOutcome(const BattleContext &state) {
    if (state.outcome == Outcome::UNDECIDED
        || state.unsupportedEffectKind != UnsupportedEffectKind::NONE)
        throw std::invalid_argument("unresolved combat has no terminal outcome label");
    // Escape is not a defeated encounter. It is reported separately by actors.
    if (state.outcome != Outcome::PLAYER_VICTORY || state.escapedCombat || state.player.curHp <= 0)
        return {};
    int inventory = 0;
    for (int i=0;i<state.potionCapacity;++i)
        inventory += state.potions[i] != Potion::EMPTY_POTION_SLOT && state.potions[i] != Potion::INVALID;
    return {1.0, state.player.curHp/100.0, inventory/5.0,
            state.player.maxHp/100.0, state.player.gold/1000.0};
}

CombatLearningSearch::CombatLearningSearch(std::vector<BattleContext> states,
        const BattleContext &source, std::uint64_t seed, CombatObjective target)
    : particles(std::move(states)), observed(source), objective(target), random(seed), rollout(source) {
    if (particles.empty() || source.outcome != Outcome::UNDECIDED)
        throw std::invalid_argument("combat learning search requires a nonterminal public belief");
    if (!std::isfinite(target.survival) || target.survival < 1
        || !std::isfinite(target.potionHp) || target.potionHp < 0 || target.potionHp > 100
        || !std::isfinite(target.maxHpPrice) || target.maxHpPrice < 0
        || !std::isfinite(target.goldHpPrice) || target.goldHpPrice < 0)
        throw std::invalid_argument("invalid combat resource objective");
    rollout.rolloutMode = 2;
    rollout.randGen.seed(seed);
    node(ROOT, particles.front());
}

CombatLearningSearch::Node &CombatLearningSearch::node(std::uint64_t key, const BattleContext &state) {
    auto found = nodes.find(key);
    if (found != nodes.end()) return *found->second;
    auto item = std::make_unique<Node>();
    for (auto action : Action::getAllActionsInState(state)) {
        if (!action.isValidAction(state)) continue;
        const auto semantic = PublicBeliefCombatSearch::publicActionKey(state, action);
        if (std::none_of(item->edges.begin(), item->edges.end(),
                        [&](const auto &e){return e.semantic == semantic;}))
            item->edges.push_back({action, semantic});
    }
    if (item->edges.empty()) throw std::runtime_error("empty learning search action support");
    std::sort(item->edges.begin(), item->edges.end(), [](const auto &a, const auto &b){return a.semantic<b.semantic;});
    for (auto &edge : item->edges) edge.prior = 1.0/item->edges.size();
    auto *result = item.get(); nodes.emplace(key, std::move(item)); return *result;
}

std::size_t CombatLearningSearch::select(const Node &at) const {
    std::size_t selected=0;
    double best=-std::numeric_limits<double>::infinity();
    // A cheap lucky win must not prevent measuring the other legal actions.
    // Prefer the neural prior's ordering, but visit every supported action
    // before repeatedly exploiting it when the node budget permits.
    bool unvisited=false;
    for(std::size_t i=0;i<at.edges.size();++i) {
        const auto &edge=at.edges[i];
        if(edge.visits==0 && edge.inFlight==0 && (!unvisited || edge.prior>best)) {
            unvisited=true;best=edge.prior;selected=i;
        }
    }
    if(unvisited)return selected;
    best=-std::numeric_limits<double>::infinity();
    for (std::size_t i=0;i<at.edges.size();++i) {
        const auto &edge=at.edges[i];
        const auto count=edge.visits+edge.inFlight;
        const double q=count ? objective.score(edge.sum)/count : 0;
        const double score=q+1.5*edge.prior*std::sqrt(1.0+at.visits+at.inFlight)/(1+count);
        if (score>best) {best=score;selected=i;}
    }
    return selected;
}

void CombatLearningSearch::release(Path &path) {
    for (auto &[at,index] : path) {--at->inFlight;--at->edges[index].inFlight;}
}

void CombatLearningSearch::backup(Path &path, const CombatOutcome &value) {
    release(path);
    for (auto &[at,index] : path) {
        ++at->visits;
        auto &edge=at->edges[index]; ++edge.visits;
        for (int i=0;i<5;++i) edge.sum[i]+=value[i];
    }
    ++simulations;
}

bool CombatLearningSearch::requestOne() {
    std::uniform_int_distribution<int> sample(0, static_cast<int>(particles.size())-1);
    BattleContext state(particles[sample(random)]);
    Path path;
    auto key=ROOT;
    for (int depth=0;depth<maximumActions;++depth) {
        if (state.unsupportedEffectKind != UnsupportedEffectKind::NONE)
            throw std::runtime_error("unsupported native effect during combat learning search");
        if (state.outcome != Outcome::UNDECIDED) {
            backup(path,terminalOutcome(state)); ++terminalEvaluations; return true;
        }
        auto &at=node(key,state);
        if (at.pending) {release(path);return false;}
        if (!at.expanded) {
            at.pending=true;
            const auto id=++nextRequest;
            pending.emplace(id,Request{id,&at,std::move(state),std::move(path)});
            return true;
        }
        const auto index=select(at);
        auto action=match(state,at.edges[index]);
        ++at.inFlight; ++at.edges[index].inFlight;
        path.emplace_back(&at,index);
        action.execute(state);
        append(key,at.edges[index].semantic);append(key,PublicBeliefCombatSearch::publicObservation(state));
    }
    // Budget censoring is counted explicitly, never exported as a death label.
    backup(path,{});++unresolvedEvaluations;return true;
}

std::vector<std::uint64_t> CombatLearningSearch::requestBatch(int count, std::int64_t budget) {
    if (count<=0 || budget<=0) throw std::invalid_argument("invalid neural leaf batch budget");
    if (!pending.empty()) throw std::logic_error("submit previous leaf batch before requesting another");
    const auto first=nextRequest;
    for (int i=0;i<count && simulations+static_cast<std::int64_t>(pending.size())<budget;++i)
        if (!requestOne()) break;
    std::vector<std::uint64_t> result;
    for (const auto &[id,request] : pending) if (id>first) result.push_back(id);
    return result;
}

std::vector<Action> CombatLearningSearch::requestActions(const Request &request) const {
    std::vector<Action> actions;
    for (const auto &edge : request.node->edges) actions.push_back(match(request.state,edge));
    return actions;
}

void CombatLearningSearch::submit(std::uint64_t id, const std::vector<double> &priors,
                                  const CombatOutcome &value) {
    auto found=pending.find(id);
    if (found==pending.end()) throw std::invalid_argument("unknown or already submitted neural leaf");
    auto &request=found->second;
    if (priors.size()!=request.node->edges.size()) throw std::invalid_argument("neural policy/action alignment mismatch");
    double total=0;
    for (auto p:priors) {if(!std::isfinite(p)||p<0) throw std::invalid_argument("invalid neural prior");total+=p;}
    if (total<=0) throw std::invalid_argument("empty neural prior support");
    for (auto v:value) if(!std::isfinite(v)||v<0) throw std::invalid_argument("invalid neural fight value");
    if (value[0]>1.000001 || value[2]>1.000001) throw std::invalid_argument("invalid probability or potion capacity");
    for (std::size_t i=0;i<priors.size();++i)
        request.node->edges[i].prior=0.97*priors[i]/total+0.03/priors.size();
    request.node->expanded=true;request.node->pending=false;
    backup(request.path,value);pending.erase(found);
}

void CombatLearningSearch::searchGuided(std::int64_t budget) {
    while(simulations<budget) {
        auto ids=requestBatch(1,budget);
        for (auto id:ids) {
            const auto &request=pending.at(id);
            const auto actions=requestActions(request);
            std::vector<double> priors;
            BattleScumSearcher2::Node temporary;
            rollout.enumerateActionsForNode(temporary,request.state);
            const auto hint=temporary.edges[rollout.selectRolloutAction(temporary,request.state)].action;
            for (auto action:actions) priors.push_back(action.bits==hint.bits ? 4.0 : 1.0);
            submitGuided(id,priors);
        }
    }
}

void CombatLearningSearch::submitGuided(std::uint64_t id,const std::vector<double> &priors) {
    BattleContext end(pending.at(id).state);
    BattleScumSearcher2::Node temporary;
    for(int step=0;step<maximumActions && end.outcome==Outcome::UNDECIDED;++step) {
        temporary.edges.clear();rollout.enumerateActionsForNode(temporary,end);
        if(temporary.edges.empty())throw std::runtime_error("empty guided learning rollout");
        temporary.edges[rollout.selectRolloutAction(temporary,end)].action.execute(end);
    }
    CombatOutcome value{};
    if(end.unsupportedEffectKind!=UnsupportedEffectKind::NONE)
        throw std::runtime_error("unsupported effect in guided learning rollout");
    if(end.outcome!=Outcome::UNDECIDED){value=terminalOutcome(end);++terminalEvaluations;}
    else ++unresolvedEvaluations;
    submit(id,priors,value);
}

const CombatLearningSearch::Node &CombatLearningSearch::root() const {return *nodes.at(ROOT);}

Action CombatLearningSearch::selectedAction() const {
    if (!pending.empty()) throw std::logic_error("cannot act while neural leaves remain pending");
    const auto &edges=root().edges;
    auto best=std::max_element(edges.begin(),edges.end(),[&](const auto &a,const auto &b){
        if(a.visits!=b.visits)return a.visits<b.visits;
        if(objective.score(a.sum)!=objective.score(b.sum))return objective.score(a.sum)<objective.score(b.sum);
        return a.prior<b.prior;
    });
    return match(observed,*best);
}
}
