

class Node:
    def __init__(self, state, parent=None):
        self._state = state
        self.parent = parent
        self._value = 0
        self._visits = 0

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

    @property
    def visits(self):
        return self._visits

    def visit(self):
        self._visits += 1

    def is_first_visit(self):
        return self._visits == 1


class DecisionNode(Node):
    def __init__(self, state, parent=None, terminal=False, quantile_idx=-1):
        super(DecisionNode, self).__init__(state, parent)
        self._terminal = terminal
        self._quantile_idx = quantile_idx
        self._prediction = None

    @property
    def terminal(self):
        return self._terminal

    @terminal.setter
    def terminal(self, terminal):
        self._terminal = terminal

    @property
    def quantile_idx(self):
        return self._quantile_idx

    @quantile_idx.setter
    def quantile_idx(self, quantile_idx):
        self._quantile_idx = quantile_idx

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = prediction

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

    def get_successor(self, quantile):
        for decision_node in self.successors:
            if decision_node.quantile_idx == quantile:
                return decision_node
        return None

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, new_reward):
        self._reward = new_reward

    @property
    def action(self):
        return self._action


