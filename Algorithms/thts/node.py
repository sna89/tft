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
    def __init__(self, state, parent=None, terminal=False):
        super(DecisionNode, self).__init__(state, parent)
        self.terminal = terminal

    def is_root(self):
        return self.parent is None

    def __eq__(self, other):
        if isinstance(other, DecisionNode):
            return self.state == other.state
        return False


class ChanceNode(Node):
    def __init__(self, state, parent=None, action=None):
        super(ChanceNode, self).__init__(state, parent)
        self._action = action
        self._reward = 0
        self._reward_group_mapping = {}

    @property
    def heuristic_value(self):
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

    @property
    def reward_group_mapping(self):
        return self._reward_group_mapping

    @reward_group_mapping.setter
    def reward_group_mapping(self, reward_group_mapping):
        self._reward_group_mapping = reward_group_mapping



