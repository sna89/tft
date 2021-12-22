from Algorithms.thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state
from env_thts_common import get_reward, build_next_state, EnvState, is_alertable_state, \
    get_group_state, get_group_names, get_num_iterations, get_last_val_time_idx
import time
import pandas as pd
from typing import Dict, List
from Algorithms.render import render
from Algorithms.statistics import init_statistics, update_statistics
from data_utils import get_group_lower_and_upper_bounds, is_group_prediction_out_of_bound, reverse_key_value_mapping
from evaluation import evaluate_classification_from_conf_matrix
from Algorithms.thts.thts_helper_functions import uct


class TrialBasedHeuristicTree:
    def __init__(self, env, config):
        self.env_name = "real"
        self.config = config
        self.env = env
        self.forecasting_model = self.env.forecasting_model
        self.group_id_group_name_mapping = self.env.group_id_group_name_mapping
        self.group_names = get_group_names(self.group_id_group_name_mapping)
        self.num_actions = self.env.action_space.n

        self.num_trials = config.get("THTS").get("NumTrials")
        self.trial_length = config.get("THTS").get("TrialLength")
        self.alert_prediction_steps = self.env.steps_from_alert
        self.consider_trial_length = True
        self.restart_env_iterations = self.env.restart_steps

    @staticmethod
    def _backup_decision_node(node: DecisionNode):
        raise NotImplementedError

    @staticmethod
    def _backup_chance_node(node: ChanceNode):
        raise NotImplementedError

    @staticmethod
    def select_chance_node_from_decision_node(node: DecisionNode):
        successor_nodes_uct_values = [uct(chance_node) for chance_node in node.successors]
        max_idx = get_argmax_from_list(successor_nodes_uct_values, choose_random=True)
        return node.successors[max_idx]

    def run(self, test_df):
        state = self.get_initial_state()
        initial_node = DecisionNode(state, parent=None)
        current_node = deepcopy(initial_node)

        action_history = []
        reward_history = []
        terminal_history = [{group_name: False for group_name in self.group_names}]
        restart_history = [{group_name: False for group_name in self.group_names}]
        steps_from_alert_history = []
        restart_steps_history = []

        num_iterations = get_num_iterations(self.config, test_df) + 1
        statistics = init_statistics(self.group_names)

        for iteration in range(1, num_iterations):
            start = time.time()
            action = self._choose_action(current_node)
            group_name_action_mapping = self._postprocess_action(action,
                                                            current_node,
                                                            terminal_history,
                                                            restart_history,
                                                            test_df,
                                                            iteration)

            next_state, env_terminal_list, restart_states, reward_group_mapping = self._transition_real_env(
                current_node,
                test_df,
                iteration,
                group_name_action_mapping)

            next_state = self._after_transition(next_state, env_terminal_list)

            action_history.append(group_name_action_mapping)
            reward_history.append(reward_group_mapping)
            terminal_history.append(env_terminal_list)
            restart_history.append(restart_states)
            steps_from_alert_history.append({group_name: group_state.steps_from_alert
                                             for group_name, group_state
                                             in current_node.state.env_state.items()})
            restart_steps_history.append({group_name: group_state.restart_steps
                                          for group_name, group_state
                                          in current_node.state.env_state.items()})

            statistics = update_statistics(self.config,
                                           self.group_names,
                                           statistics,
                                           reward_group_mapping,
                                           steps_from_alert_history[-1])

            render(self.config,
                   self.group_names,
                   test_df,
                   action_history,
                   reward_history,
                   terminal_history,
                   restart_history,
                   steps_from_alert_history,
                   restart_steps_history)

            current_node = DecisionNode(next_state, parent=current_node, terminal=False)

            print(iteration)
            end = time.time()
            run_time = end - start
            print(run_time)

        print("Evaluation")
        for group_name, group_statistics in statistics.items():
            print(group_name)
            tp, fn, fp, tn = list(group_statistics.values())
            evaluate_classification_from_conf_matrix(tn, fp, fn, tp)

        print("Cumulative Reward")
        cumulative_reward_dict = {}
        for reward in reward_history:
            for group_name, group_reward in reward.items():
                if group_name not in cumulative_reward_dict:
                    cumulative_reward_dict[group_name] = 0

                cumulative_reward_dict[group_name] += group_reward
        print(cumulative_reward_dict)

    def get_initial_state(self):
        self.env.reset()
        initial_state = self.env.current_state
        return initial_state

    def _choose_action(self, current_node: DecisionNode):
        action = 0
        if not is_alertable_state(current_node, self.alert_prediction_steps, self.restart_env_iterations):
            pass
        else:
            for trial in range(self.num_trials):
                self._run_trial(current_node)
            action = self.select_max_value_action(current_node)
        return action

    def _postprocess_action(self,
                            action: int,
                            current_decision_node: DecisionNode,
                            terminal_history: List[Dict],
                            restart_history: List[Dict],
                            test_df: pd.DataFrame(),
                            iteration: int
                            ) -> Dict:
        if action == 0:
            return {group_name: action for group_name in self.group_names}
        else:
            action_dict = {}
            child_chance_node = current_decision_node.successors[-1]
            child_decision_nodes = child_chance_node.successors
            for child_decision_node in child_decision_nodes:
                grand_child_chance_node = child_decision_node.successors[0]
                for group_name, reward in grand_child_chance_node.reward_group_mapping.items():
                    if reward > 0:
                        if group_name not in action_dict:
                            action_dict[group_name] = 1
                        else:
                            action_dict[group_name] += 1

            for group_name in self.group_names:
                is_restart = terminal_history[-1][group_name]
                is_terminal = restart_history[-1][group_name]
                group_state = get_group_state(current_decision_node.state.env_state, group_name)
                if not is_restart \
                        and not is_terminal \
                        and group_state.steps_from_alert == self.env.steps_from_alert:
                    if group_name not in action_dict:
                        action_dict[group_name] = 0
                    else:
                        action_dict[group_name] = 1
                else:
                    action_dict[group_name] = 0
            return action_dict

    def get_encoder_time_idx_range(self, test_df: pd.DataFrame(), iteration: int):
        min_time_idx = test_df.time_idx.min()
        time_idx_range_start = min_time_idx + iteration - 1
        time_idx_range_end = time_idx_range_start + self.config.get("EncoderLength") + self.config.get(
            "PredictionLength")
        time_idx_range = list(range(time_idx_range_start, time_idx_range_end))
        return time_idx_range

    def group_prediction_exceed_bounds(self, group_prediction, group_name: str):
        lb, ub = get_group_lower_and_upper_bounds(self.config, group_name)
        out_of_bound, out_of_bound_idx = is_group_prediction_out_of_bound(group_prediction, lb, ub)
        return out_of_bound, out_of_bound_idx

    def _after_transition(self, next_state: EnvState, is_terminal_list: dict):
        for group_name, group_state in next_state.env_state.items():
            is_group_current_terminal = is_terminal_list[group_state.group]
            group_restart_steps = group_state.restart_steps
            if self.is_group_restart(is_group_current_terminal, group_restart_steps):
                next_state.env_state[group_name].steps_from_alert = self.env.steps_from_alert

        return next_state

    def is_group_restart(self, is_group_current_terminal, group_restart_steps):
        if is_group_current_terminal or group_restart_steps < self.restart_env_iterations:
            return True
        return False

    def _run_trial(self, root_node: DecisionNode):
        set_env_to_state(self.env, root_node.state)
        self._visit_decision_node(root_node, depth=0, terminal=False)

    def _visit_decision_node(self, decision_node: DecisionNode, depth: int, terminal: bool = False):
        if not terminal:
            decision_node.visit()
            if decision_node.is_first_visit():
                self._initialize_decision_node(decision_node)  # expansion
            chance_node = self.select_chance_node_from_decision_node(decision_node)  # select action
            self._visit_chance_node(chance_node, depth)
        self._backup_decision_node(decision_node)

    def _visit_chance_node(self, chance_node: ChanceNode, depth: int):
        terminal = False
        chance_node.visit()
        next_state = self._select_outcome(chance_node)
        if self.consider_trial_length and depth == self.trial_length - 1:
            terminal = True

        decision_node = self.add_decision_node(next_state, chance_node, terminal=terminal)
        self._visit_decision_node(decision_node, depth + 1, terminal)
        self._backup_chance_node(chance_node)

    def _select_outcome(self, chance_node: ChanceNode):
        set_env_to_state(self.env, chance_node.state)
        next_state, reward_group_mapping = self.env.step(chance_node.action)  # monte carlo sample
        reward = self._aggregate_reward(reward_group_mapping)
        chance_node.reward = reward
        chance_node.reward_group_mapping = reward_group_mapping
        return next_state

    def _initialize_decision_node(self, decision_node: DecisionNode):
        feasible_actions = self._get_feasible_actions_for_node(decision_node)
        for action in feasible_actions:
            self.add_chance_node(decision_node, action=action)

    def _get_feasible_actions_for_node(self, decision_node: DecisionNode):
        actions = list(range(self.num_actions))
        if not is_alertable_state(decision_node, self.alert_prediction_steps, self.restart_env_iterations):
            actions.remove(1)
        return actions

    @staticmethod
    def _backup_decision_node(decision_node: DecisionNode):
        raise NotImplementedError

    @staticmethod
    def _backup_chance_node(chance_node: ChanceNode):
        raise NotImplementedError

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
    def add_decision_node(next_state, chance_node: ChanceNode, terminal: bool = False):
        decision_node = TrialBasedHeuristicTree._get_decision_node_from_chance_node(chance_node, next_state)
        if not decision_node:
            decision_node = DecisionNode(state=next_state, parent=chance_node, terminal=terminal)
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

    def _transition_real_env(self, node: DecisionNode, test_df: pd.DataFrame(), iteration: int, action_dict: Dict):
        val_max_time_idx = get_last_val_time_idx(self.config, test_df)
        next_sample = test_df[lambda x: x.time_idx == (val_max_time_idx + iteration)]
        next_sample.set_index(self.config.get("GroupKeyword"), inplace=True)
        next_state_values = next_sample[[self.config.get("ValueKeyword")]].to_dict(orient="dict")[
            self.config.get("ValueKeyword")]

        current_state = node.state
        next_state, next_state_terminal, next_state_restart = build_next_state(self.env_name,
                                                                               self.config,
                                                                               current_state,
                                                                               self.group_names,
                                                                               next_state_values,
                                                                               self.env.steps_from_alert,
                                                                               self.env.restart_steps,
                                                                               action_dict)

        reward_group_mapping = get_reward(self.env_name,
                                          self.config,
                                          self.group_names,
                                          next_state_terminal,
                                          current_state,
                                          action_dict)

        return next_state, next_state_terminal, next_state_restart, reward_group_mapping

    @staticmethod
    def _aggregate_reward(reward_group_mapping):
        return sum([reward for group_name, reward in reward_group_mapping.items()]) / float(len(reward_group_mapping))