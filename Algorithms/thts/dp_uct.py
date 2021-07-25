from Algorithms.thts import TrialBasedHeuristicTree
from Algorithms.thts import DecisionNode, ChanceNode


class DpUCT(TrialBasedHeuristicTree):
    def __init__(self, env, config):
        super(DpUCT, self).__init__(env, config)

    @staticmethod
    def _backup_decision_node(decision_node: DecisionNode):
        decision_node.backup_dp_uct()

    @staticmethod
    def _backup_chance_node(chance_node: ChanceNode):
        chance_node.backup_dp_uct()
