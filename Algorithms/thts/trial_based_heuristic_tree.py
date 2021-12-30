from Algorithms.thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from data_utils import reverse_key_value_mapping
from utils import get_argmax_from_list
from EnvCommon.env_thts_common import get_reward_from_env, build_next_state, EnvState, \
    get_group_state, get_num_iterations, get_last_val_time_idx, set_env_state, set_env_group, \
    get_reward_for_alert_from_prediction
import time
import pandas as pd
import numpy as np
from Algorithms.render import render
from Algorithms.statistics import init_statistics, update_statistics
from evaluation import evaluate_classification_from_conf_matrix
from Algorithms.thts.thts_helper_functions import uct, run_heuristic


class TrialBasedHeuristicTree:
    def __init__(self, config, env, predictor, group_name):
        self.env_name = "real"
        self.config = config
        self.env = env
        self.predictor = predictor
        self.tree_group_name = group_name

        self.group_id_group_name_mapping = self.env.group_id_group_name_mapping
        self.group_name_group_id_mapping = reverse_key_value_mapping(self.group_id_group_name_mapping)
        self.tree_group_id = self.group_name_group_id_mapping[self.tree_group_name]

        self.group_names = self.env.group_names
        self.num_actions = self.env.action_space

        self.num_trials = config.get("THTS").get("NumTrials")
        self.trial_length = config.get("THTS").get("TrialLength")

        self.consider_trial_length = True
        self.val_max_time_idx = None

    @staticmethod
    def _backup_decision_node(node: DecisionNode):
        raise NotImplementedError

    @staticmethod
    def _backup_chance_node(node: ChanceNode):
        raise NotImplementedError

    def set_env_group_name(self):
        self.env.env_group_name = self.tree_group_name

    def run(self, test_df):
        state = self.get_initial_state()
        initial_node = DecisionNode(state, parent=None)
        current_node = deepcopy(initial_node)

        action_history = []
        reward_history = []
        terminal_history = [False]
        restart_history = [False]
        steps_from_alert_history = []
        restart_steps_history = []

        self.val_max_time_idx = get_last_val_time_idx(self.config, test_df)

        num_iterations = get_num_iterations(self.config, test_df) + 1
        statistics = init_statistics(self.tree_group_name)

        for iteration in range(1, num_iterations):
            start = time.time()

            action = self._choose_action(current_node)

            next_state, is_next_state_terminal, is_next_state_restart, reward = self._transition_real_env(
                current_node,
                test_df,
                iteration,
                action)

            next_state = self._after_transition(next_state, is_next_state_terminal)

            action_history.append(action)
            reward_history.append(reward)
            terminal_history.append(is_next_state_terminal)
            restart_history.append(is_next_state_restart)
            steps_from_alert_history.append(current_node.state.env_state[self.tree_group_name].steps_from_alert)
            restart_steps_history.append(current_node.state.env_state[self.tree_group_name].restart_steps)

            statistics = update_statistics(self.config,
                                           self.tree_group_name,
                                           statistics,
                                           reward,
                                           steps_from_alert_history[-1])

            render(self.config,
                   self.tree_group_name,
                   test_df,
                   action_history,
                   reward_history,
                   terminal_history,
                   restart_history,
                   steps_from_alert_history,
                   restart_steps_history)

            current_node = DecisionNode(next_state, parent=current_node, terminal=False)

            print()
            print("Group Name: {}, Iteration: {}, Action: {}, Reward: {}".format(self.tree_group_name,
                                                                                 iteration,
                                                                                 action,
                                                                                 reward))
            print()
            end = time.time()
            run_time = end - start
            print("Group Name: {}, RunTime: {}".format(self.tree_group_name, run_time))

        print("Group Name: {}, Cumulative Reward: {}".format(self.tree_group_name, sum(reward_history)))
        print("Group Name: {} Evaluation:")
        print(["="] * 30)

        for group_name, group_statistics in statistics.items():
            if group_name == self.tree_group_name:
                tp, fn, fp, tn = list(group_statistics.values())
                evaluate_classification_from_conf_matrix(tn, fp, fn, tp)

    def select_chance_node_from_decision_node(self, decision_node: DecisionNode):
        if len(decision_node.successors) == 1:
            return decision_node.successors[0]

        if all([chance_node.visits == 0 for chance_node in decision_node.successors]):
            list(map(lambda chance_node: run_heuristic(self.config,
                                                       chance_node,
                                                       decision_node.prediction,
                                                       self.tree_group_name),
                     decision_node.successors))
            max_idx = get_argmax_from_list([chance_node.value for chance_node in decision_node.successors])
        else:
            successor_nodes_uct_values = [uct(chance_node) for chance_node in decision_node.successors]
            max_idx = get_argmax_from_list(successor_nodes_uct_values, choose_random=True)

        return decision_node.successors[max_idx]

    @staticmethod
    def select_decision_node_from_chance_node(node: ChanceNode):
        num_successors = len(node.successors)
        chosen_idx = np.random.choice(list(range(num_successors)))
        return node.successors[chosen_idx]

    def get_initial_state(self):
        self.env.reset()
        initial_state = self.env.current_state
        return initial_state

    def _choose_action(self, current_node: DecisionNode):
        action = 0
        if self.is_alertable_state(current_node,
                                   self.env.env_steps_from_alert,
                                   self.env.env_restart_steps,
                                   self.tree_group_name):
            for _ in range(self.num_trials):
                self._run_trial(current_node)
            action = self.select_max_value_action(current_node)
        return action

    def _after_transition(self, next_state: EnvState, is_current_terminal: dict):
        group_state = get_group_state(next_state.env_state, self.tree_group_name)
        if self.is_group_next_state_restart(is_current_terminal, group_state.restart_steps):
            next_state.env_state[self.tree_group_name].steps_from_alert = self.env.env_steps_from_alert

        return next_state

    def is_group_next_state_restart(self, is_group_current_terminal, group_restart_steps):
        if is_group_current_terminal or (1 < group_restart_steps < self.env.env_restart_steps):
            return True
        return False

    def _run_trial(self, root_node: DecisionNode):
        set_env_state(self.env, root_node.state)
        self._visit_decision_node(root_node, depth=0, terminal=False)

    def _visit_decision_node(self, decision_node: DecisionNode, depth: int, terminal: bool = False):
        decision_node.visit()
        if not terminal:
            if decision_node.is_first_visit():
                self._expand_decision_node(decision_node)  # expansion
                prediction = self._run_prediction(decision_node)
                decision_node.prediction = prediction
            else:
                prediction = decision_node.prediction

            chance_node = self.select_chance_node_from_decision_node(decision_node)  # select action
            self._visit_chance_node(chance_node, prediction, depth)

        self._backup_decision_node(decision_node)

    def _visit_chance_node(self, chance_node: ChanceNode, prediction, depth: int):
        terminal = False
        if self.consider_trial_length and depth == self.trial_length - 1:
            terminal = True

        if chance_node.action == 0:
            chance_node.visit()

            chosen_quantile = self._draw_decision_node_quantile()
            decision_node = chance_node.get_successor(chosen_quantile)

            if chance_node.is_first_visit() or not decision_node:
                sampled_prediction = self.predictor.sample_from_prediction(prediction, self.tree_group_id,
                                                                           chosen_quantile)
                next_state, reward = self._create_next_state(chance_node, sampled_prediction)
                chance_node.reward = reward
                decision_node = self.add_decision_node(next_state,
                                                       chance_node,
                                                       chosen_quantile=chosen_quantile,
                                                       terminal=terminal)
            else:
                decision_node.terminal = terminal
            self._visit_decision_node(decision_node, depth + 1, terminal)

        elif chance_node.action == 1:
            reward = get_reward_for_alert_from_prediction(self.config,
                                                          self.tree_group_name,
                                                          prediction,
                                                          self.env.env_steps_from_alert)
            chance_node.reward = reward

        self._backup_chance_node(chance_node)

    def _create_next_state(self, chance_node: ChanceNode, prediction):
        set_env_state(self.env, chance_node.state)
        set_env_group(self.env, self.tree_group_name)

        next_state, reward = self.env.step(chance_node.action, prediction)  # monte carlo sample
        return next_state, reward

    def _expand_decision_node(self, decision_node: DecisionNode):
        feasible_actions = self._get_feasible_actions_for_node(decision_node)
        for action in feasible_actions:
            self.add_chance_node(decision_node, action=action)

    def _get_feasible_actions_for_node(self, decision_node: DecisionNode):
        actions = list(range(self.num_actions))
        # if not self.is_alertable_state(decision_node,
        #                                self.env.env_steps_from_alert,
        #                                self.env.env_restart_steps,
        #                                self.tree_group_name):
        #     actions.remove(1)
        return actions

    @staticmethod
    def select_max_value_action(decision_node: DecisionNode):
        successor_values = [successor_node.value for successor_node in decision_node.successors]
        argmax_successor = get_argmax_from_list(successor_values, choose_random=True)
        greedy_action = decision_node.successors[argmax_successor].action
        return greedy_action

    @staticmethod
    def add_chance_node(decision_node: DecisionNode, action: int):
        chance_node = ChanceNode(state=decision_node.state,
                                 parent=decision_node,
                                 action=action)
        decision_node.add_successor(chance_node)

    @staticmethod
    def add_decision_node(next_state,
                          chance_node: ChanceNode,
                          chosen_quantile: int,
                          terminal: bool = False):
        decision_node = DecisionNode(state=next_state,
                                     parent=chance_node,
                                     terminal=terminal,
                                     quantile_idx=chosen_quantile)
        chance_node.add_successor(decision_node)
        return decision_node

    @staticmethod
    def _get_decision_node_from_chance_node(chance_node, next_state):
        if TrialBasedHeuristicTree._decision_node_exists(chance_node, next_state):
            for successor in chance_node.successors:
                if successor.state == next_state:
                    return successor
        else:
            return None

    @staticmethod
    def _decision_node_exists(chance_node, next_state):
        for successor in chance_node.successors:
            if successor.state == next_state:
                return True
        return False

    def _transition_real_env(self, node: DecisionNode, test_df: pd.DataFrame(), iteration: int, action: int):
        next_sample = test_df[lambda x: x.time_idx == (self.val_max_time_idx + iteration)]
        next_sample.set_index(self.config.get("GroupKeyword"), inplace=True)
        next_state_values = next_sample[[self.config.get("ValueKeyword")]].to_dict(orient="dict")[
            self.config.get("ValueKeyword")]

        current_state = node.state
        next_state, next_state_terminal, next_state_restart = build_next_state(self.env_name,
                                                                               self.config,
                                                                               current_state,
                                                                               self.tree_group_name,
                                                                               self.group_names,
                                                                               next_state_values,
                                                                               self.env.env_steps_from_alert,
                                                                               self.env.env_restart_steps,
                                                                               action)

        reward = get_reward_from_env(self.env_name,
                                     self.config,
                                     self.tree_group_name,
                                     next_state_terminal,
                                     current_state,
                                     self.env.env_steps_from_alert,
                                     self.env.env_restart_steps,
                                     action)

        return next_state, next_state_terminal, next_state_restart, reward

    def _draw_decision_node_quantile(self):
        num_quantiles = self.predictor.num_quantiles
        chosen_quantile = np.random.choice(list(range(1, num_quantiles - 1)))
        return chosen_quantile

    @staticmethod
    def _is_successor_decision_node_exists(chance_node: ChanceNode, chosen_quantile: int):
        for successor_decision_node in chance_node.successors:
            if successor_decision_node.quantile_idx == chosen_quantile:
                return True
        return False

    @staticmethod
    def is_alertable_state(current_node: DecisionNode, env_steps_from_alert: int, env_restart_steps: int, group_name):
        group_state = current_node.state.env_state[group_name]
        if group_state.steps_from_alert < env_steps_from_alert or group_state.restart_steps < env_restart_steps:
            return False
        return True

    def _run_prediction(self, decision_node: DecisionNode):
        prediction = self.predictor.predict(decision_node.state)
        return prediction
