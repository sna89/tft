from Algorithms.thts.trial_based_heuristic_tree import TrialBasedHeuristicTree
from Algorithms.thts.thts_helper_functions import backup_decision_full_bellman, backup_chance_monte_carlo
from Algorithms.thts.node import ChanceNode, DecisionNode


class MaxUCT(TrialBasedHeuristicTree):
    def __init__(self, env, config):
        super(MaxUCT, self).__init__(env, config)

    @staticmethod
    def _backup_decision_node(node: DecisionNode):
        backup_decision_full_bellman(node)

    @staticmethod
    def _backup_chance_node(node: ChanceNode):
        backup_chance_monte_carlo(node)



