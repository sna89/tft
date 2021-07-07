from thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state, get_node_expandable_actions
import pandas as pd


class TrialBasedHeuristicTree:
    def __init__(self, env, config):
        self.simulator_env = env
        self.real_env = deepcopy(env)
        self.num_actions = self.simulator_env.action_space.n

        self.env_name = config.get("env_name")
        self.num_trials = config.get("THTS").get("NumTrials")
        self.trial_length = config.get("THTS").get("TrialLength")
        self.uct_bias = config.get("THTS").get("UCTBias")
        self.runs = config.get("THTS").get("Runs")
        self.consider_trial_length = True

    def get_initial_state(self):
        initial_state = self.simulator_env.reset()
        return initial_state

    def run(self):
        state = self.get_initial_state()
        initial_node = DecisionNode(state, parent=None)

        for run in range(1, self.runs + 1):
            node = deepcopy(initial_node)
            set_env_to_state(self.real_env, node.state)
            tot_reward = 0
            while True:
                for trial in range(self.num_trials):
                    self._run_trial(node)
                action = self.select_greedy_action(node)
                next_state, reward, terminal, _ = self.real_env.step(action)
                tot_reward += reward

                if terminal:
                    print(tot_reward)
                    break
                else:
                    node = DecisionNode(next_state, parent=node, terminal=False)

    def _run_trial(self, root_node: DecisionNode):
        depth = 0
        set_env_to_state(self.simulator_env, root_node.state)
        self._visit_decision_node(root_node, depth, False)

    def _visit_decision_node(self, decision_node: DecisionNode, depth: int, terminal: bool = False):
        if not terminal:
            decision_node.visit()
            if decision_node.is_first_visit():
                self._initialize_decision_node(decision_node)  # expansion
            chance_node = decision_node.select_chance_node()  # select action
            self._visit_chance_node(chance_node, depth)
        self._backup_decision_node(decision_node)

    def _visit_chance_node(self, chance_node: ChanceNode, depth: int):
        chance_node.visit()
        next_state, terminal, prob = self._select_outcome(chance_node)
        if self.consider_trial_length and depth == self.trial_length - 1:
            terminal = True

        #check if node already exists
        decision_node = self.add_decision_node(next_state, chance_node, terminal=terminal, prob=prob)
        self._visit_decision_node(decision_node, depth + 1, terminal)
        self._backup_chance_node(chance_node)

    def _select_outcome(self, chance_node: ChanceNode):
        set_env_to_state(self.simulator_env, chance_node.state)
        next_state, reward, terminal, prob = self.simulator_env.step(chance_node.action)  # monte carlo sample
        chance_node.reward = reward
        return next_state, terminal, prob

    def _initialize_decision_node(self, decision_node: DecisionNode):
        feasible_actions = self._get_feasible_actions_for_node(decision_node)
        for action in feasible_actions:
            self.add_chance_node(decision_node, action=action)

    def _get_feasible_actions_for_node(self, decision_node: DecisionNode):
        actions = list(range(self.simulator_env.action_space.n))
        steps_from_alert = decision_node.state.steps_from_alert
        if self.simulator_env.min_steps_from_alert <= steps_from_alert < self.simulator_env.max_steps_from_alert:
            actions.remove(1)
        return actions

    @staticmethod
    def _backup_decision_node(decision_node: DecisionNode):
        raise NotImplementedError

    @staticmethod
    def _backup_chance_node(chance_node: ChanceNode):
        raise NotImplementedError

    @staticmethod
    def select_greedy_action(decision_node: DecisionNode):
        successor_values = [successor_node.value for successor_node in decision_node.successors]
        argmax_successor = get_argmax_from_list(successor_values, choose_random=True)
        greedy_action = decision_node.successors[argmax_successor].action
        return greedy_action

    def add_chance_node(self, decision_node: DecisionNode, action: int):
        chance_node = ChanceNode(state=decision_node.state,
                                 parent=decision_node,
                                 action=action,
                                 uct_bias=self.uct_bias)
        decision_node.add_successor(chance_node)

    @staticmethod
    def add_decision_node(next_state, chance_node: ChanceNode, terminal: bool = False, prob: float = 1):
        if isinstance(prob, dict):
            prob = prob['prob']
        decision_node = DecisionNode(state=next_state, parent=chance_node, terminal=terminal, prob=prob)
        chance_node.add_successor(decision_node)
        return decision_node


