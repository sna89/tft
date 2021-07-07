from gym_ad.algorithms.thts.trial_based_heuristic_tree import TrialBasedHeuristicTree
from gym_ad.node import DecisionNode, ChanceNode


class MaxUCT(TrialBasedHeuristicTree):
    def __init__(self, env, config):
        super(MaxUCT, self).__init__(env, config)

    @staticmethod
    def _backup_decision_node(decision_node: DecisionNode):
        decision_node.backup_max_uct()

    @staticmethod
    def _backup_chance_node(chance_node: ChanceNode):
        chance_node.backup_max_uct()
