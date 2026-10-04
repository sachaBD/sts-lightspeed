#include "persistence_boss_fixture.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <iostream>
#include <stdexcept>
using namespace sts;
using Search = search::PublicBeliefCombatSearch;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    auto live = bossFixture(MonsterEncounter::CHAMP, 7);
    Search tree({live}, 42, 2);
    tree.search(4096);
    const auto action = tree.selectedAction();
    const auto semantic = tree.actionKey(live, action);
    action.execute(live);
    tree.rebase({live}, semantic, 43);
    check(tree.retainedVisits > 0, "committed continuation did not retain evidence");
    const auto retained = tree.retainedVisits;
    check(tree.simulations == 0 && tree.treeActions == 0, "per-decision counters were not reset");
    tree.search(512);
    check(tree.simulations == 512, "warm decision must spend the full fresh work budget");
    check(tree.root().visits == retained + 512, "retained evidence or fresh work lost");
    check(tree.selectedAction().isValidAction(live), "warm action invalid");
    std::cout << "original history-tree continuation PASS: retained=" << retained
              << " nodes=" << tree.nodes.size() << " max_depth=" << tree.maximumTreeDepth << '\n';
}
