from Algorithms.thts.node import DecisionNode, ChanceNode
import numpy as np

UCT_BIAS = np.sqrt(2)


def uct(node: ChanceNode):
    if node.visits == 0:
        uct_value = node.heuristic_value
    else:
        uct_value = UCT_BIAS * np.sqrt(np.log(node.parent.visits) / node.visits) + node.value
    return uct_value


def backup_decision_full_bellman(node: DecisionNode):
    value = 0
    if not node.terminal:
        value = max([node.value for node in node.successors])
    node.value = value


def backup_chance_monte_carlo(node: ChanceNode):
    nominator = 0
    for decision_node in node.successors:
        nominator += decision_node.visits * decision_node.value

    denominator = node.visits
    q_value = node.reward + nominator / float(denominator)
    node.value = q_value


def backup_chance_partial_bellman(node: ChanceNode):
    nominator = 0
    denominator = 0
    for decision_node in node.successors:
        decision_node_prob = decision_node.prob
        nominator += decision_node_prob * decision_node.value
        denominator += decision_node_prob

    q_value = node.reward + nominator / float(denominator)
    node.value = q_value
