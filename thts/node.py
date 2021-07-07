from utils import get_argmax_from_list
import numpy as np


class Node:
    def __init__(self, state, parent=None):
        self._state = state
        self.parent = parent
        self._value = 0
        self.visits = 0

        self._successors = []

    @property
    def successors(self):
        return self._successors

    def add_successor(self, successor):
        self.successors.append(successor)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def visit(self):
        self.visits += 1

    def is_first_visit(self):
        return self.visits == 1


class DecisionNode(Node):
    def __init__(self, state, parent=None, terminal=False, prob=1):
        super(DecisionNode, self).__init__(state, parent)
        self.terminal = terminal
        self.prob = prob

    def select_chance_node(self):
        successor_nodes_uct_values = [chance_node.uct for chance_node in self.successors]
        max_idx = get_argmax_from_list(successor_nodes_uct_values, choose_random=True)
        return self.successors[max_idx]

    def is_root(self):
        return self.parent is None

    def backup_max_uct(self):
        value = 0
        if not self.terminal:
            value = max([node.value for node in self.successors])
        self.value = value

    def backup_dp_uct(self):
        self.backup_max_uct()


class ChanceNode(Node):
    def __init__(self, state, parent=None, action=None, uct_bias=0):
        super(ChanceNode, self).__init__(state, parent)
        self._action = action
        self._reward = 0
        self.uct_bias = uct_bias

    @property
    def uct(self):
        if self.visits == 0:
            uct_value = self.calc_heuristic()
        else:
            uct_value = self.uct_bias * np.sqrt(np.log(self.parent.visits) / self.visits) + self.value
        return uct_value

    def calc_heuristic(self):
        return np.Inf

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, new_reward):
        self._reward = new_reward

    @property
    def action(self):
        return self._action

    def backup_max_uct(self):
        nominator = 0
        for decision_node in self.successors:
            nominator += decision_node.visits * decision_node.value

        denominator = self.visits
        q_value = self.reward + nominator / float(denominator)
        self.value = q_value

    def backup_dp_uct(self):
        nominator = 0
        sum_explicit_tree_prob = 0
        for decision_node in self.successors:
            nominator += decision_node.prob * decision_node.value
            sum_explicit_tree_prob += decision_node.prob

        q_value = self.reward + nominator / sum_explicit_tree_prob
        self.value = q_value

