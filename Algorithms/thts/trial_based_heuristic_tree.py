from Algorithms.thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state
from env_thts_common import get_reward, build_next_state, EnvState, is_alertable_state, \
    get_group_lower_and_upper_bounds, get_group_state, get_group_names, get_num_iterations, \
    is_group_prediction_out_of_bound
import time
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import torch


class TrialBasedHeuristicTree:
    def __init__(self, env, config):
        self.env_name = "real"
        self.env = env
        self.model = self.env.model
        self.group_idx_mapping = self.env.group_idx_mapping
        self.group_names = get_group_names(self.group_idx_mapping)
        self.num_actions = self.env.action_space.n
        self.config = config

        self.num_trials = config.get("THTS").get("NumTrials")
        self.trial_length = config.get("THTS").get("TrialLength")
        self.uct_bias = config.get("THTS").get("UCTBias")
        self.runs = config.get("THTS").get("Runs")
        self.alert_prediction_steps = self.env.max_steps_from_alert
        self.consider_trial_length = True
        self.restart_env_iterations = self.env.max_restart_steps

    def get_initial_state(self):
        initial_state = self.env.reset()
        return initial_state

    def run(self, test_df):
        state = self.get_initial_state()
        initial_node = DecisionNode(state, parent=None)

        for run in range(1, self.runs + 1):
            current_node = deepcopy(initial_node)

            action_history = []
            reward_history = []
            terminal_history = [{group_name: False for group_name in self.group_names}]
            restart_history = [{group_name: False for group_name in self.group_names}]
            alert_prediction_steps_history = []
            restart_steps_history = []

            num_iterations = get_num_iterations(test_df, self.env.model_enc_len)
            for iteration in range(1, num_iterations):
                start = time.time()
                action, run_time = self._before_transition(current_node)
                action_dict = self._postprocess_action(action,
                                                       current_node,
                                                       terminal_history,
                                                       restart_history,
                                                       test_df,
                                                       iteration)

                next_state, env_terminal_list, restart_states, reward = self._transition_real_env(current_node,
                                                                                                  test_df,
                                                                                                  iteration,
                                                                                                  action_dict)

                next_state = self._after_transition(next_state, env_terminal_list)

                action_history.append(action_dict)
                reward_history.append(reward)
                terminal_history.append(env_terminal_list)
                restart_history.append(restart_states)
                alert_prediction_steps_history.append({group_state.series: group_state.steps_from_alert
                                                       for group_state
                                                       in current_node.state.env_state})
                restart_steps_history.append({group_state.series: group_state.restart_steps
                                                       for group_state
                                                       in current_node.state.env_state})

                end = time.time()
                run_time = end - start

                self.render(test_df,
                            run_time,
                            action_history,
                            reward_history,
                            terminal_history,
                            restart_history,
                            alert_prediction_steps_history,
                            restart_steps_history)

                current_node = DecisionNode(next_state, parent=current_node, terminal=False)

    def _before_transition(self, current_node: DecisionNode):
        action = 0
        run_time = 0
        if not is_alertable_state(current_node, self.alert_prediction_steps, self.restart_env_iterations):
            pass
        else:
            for trial in range(self.num_trials):
                self._run_trial(current_node)
            action = self.select_greedy_action(current_node)
        return action, run_time

    def _postprocess_action(self,
                            action: int,
                            current_node: DecisionNode,
                            terminal_history: List[Dict],
                            restart_history: List[Dict],
                            test_df: pd.DataFrame(),
                            iteration: int
                            ) -> Dict:
        if action == 0:
            return {group_name: action for group_name in self.group_names}
        else:
            action_dict = {}
            time_idx_range = self.get_encoder_time_idx_range(test_df, iteration)
            prediction_df = test_df[test_df.time_idx.isin(time_idx_range)]
            prediction, x = self.model.predict(prediction_df, mode="prediction", return_x=True)
            for group_name in self.group_names:
                is_restart = terminal_history[-1][group_name]
                is_terminal = restart_history[-1][group_name]
                group_state = get_group_state(current_node.state.env_state, group_name)
                if not is_restart \
                        and not is_terminal \
                        and group_state.steps_from_alert == self.env.max_steps_from_alert:
                    group_prediction = prediction[self.group_idx_mapping[group_name]]
                    if self.group_prediction_exceed_bounds(group_prediction, group_name):
                        action_dict[group_name] = 1
                    else:
                        action_dict[group_name] = 0
                else:
                    action_dict[group_name] = 0
            return action_dict

    def get_encoder_time_idx_range(self, test_df: pd.DataFrame(), iteration: int):
        min_time_idx = test_df.time_idx.min()
        time_idx_range_start = min_time_idx + iteration - 1
        time_idx_range_end = time_idx_range_start + self.env.model_enc_len + self.env.model_pred_len
        time_idx_range = list(range(time_idx_range_start, time_idx_range_end))
        return time_idx_range

    def group_prediction_exceed_bounds(self, group_prediction, group_name: str):
        lb, ub = get_group_lower_and_upper_bounds(self.config, group_name)
        return is_group_prediction_out_of_bound(group_prediction, lb, ub)

    def _after_transition(self, next_state: EnvState, env_terminal_list: dict):
        for idx, group_state in enumerate(next_state.env_state):
            env_terminal = env_terminal_list[group_state.series]
            restart_steps = group_state.restart_steps
            if self.is_restart_steps(env_terminal, restart_steps):
                next_state.env_state[idx].steps_from_alert = self.env.max_steps_from_alert

        return next_state

    def is_restart_steps(self, env_terminal, restart_env_iterations):
        if env_terminal or restart_env_iterations < self.restart_env_iterations:
            return True
        return False

    def _run_trial(self, root_node: DecisionNode):
        depth = 0
        set_env_to_state(self.env, root_node.state)
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
        next_state, reward = self.env.step(chance_node.action)  # monte carlo sample
        chance_node.reward = reward
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
        val_max_time_idx = test_df.time_idx.min() + self.config.get("EncoderLength") - 1
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
                                                                               self.env.max_steps_from_alert,
                                                                               self.env.max_restart_steps,
                                                                               action_dict)

        reward = get_reward(self.env_name,
                            self.config,
                            self.group_names,
                            next_state_terminal,
                            current_state,
                            action_dict)

        return next_state, next_state_terminal, next_state_restart, reward

    def render(self,
               test_df: pd.DataFrame(),
               run_time: float,
               action_history: List[Dict],
               reward_history: List[float],
               terminal_history: List[Dict],
               restart_history: List[Dict],
               alert_prediction_steps_history: List[Dict],
               restart_steps_history: List[Dict]):

        print("Action: {}".format(action_history[-1]))
        print("Reward: {}".format(reward_history[-1]))
        print("Iteration RunTime: {}".format(run_time))

        min_test_time_idx = self._get_min_test_time_idx(test_df)
        time_idx_list = list(test_df[test_df.time_idx >= min_test_time_idx]['time_idx'].unique())

        for group_name in self.group_names:
            fig = go.Figure()
            fig = self._add_group_y_value_plot(fig, test_df, group_name, time_idx_list)

            current_time_idx_list = list(time_idx_list[:len(action_history)])
            for idx, time_idx in enumerate(current_time_idx_list):
                reward = reward_history[idx]
                action = action_history[idx][group_name]
                terminal = terminal_history[idx][group_name]
                restart = restart_history[idx][group_name]
                steps_from_alert = alert_prediction_steps_history[idx][group_name]
                restart_steps = restart_steps_history[idx][group_name]

                fig = self._add_group_step_decision_to_plot(fig,
                                                            test_df,
                                                            group_name,
                                                            time_idx,
                                                            reward,
                                                            action,
                                                            terminal,
                                                            restart,
                                                            steps_from_alert,
                                                            restart_steps)

            fig.update_xaxes(title_text="<b>time_idx</b>")
            fig.update_yaxes(title_text="<b>Actual</b>")
            fig.write_html('render_synthetic_{}.html'.format(group_name))

    def _add_group_y_value_plot(self, fig, test_df, group_name, time_idx_list):
        group_y = list(test_df[(test_df[self.config.get("GroupKeyword")] == group_name)
                               & (test_df.time_idx.isin(time_idx_list))]
                       [self.config.get("ValueKeyword")].values)
        fig.add_trace(
            go.Scatter(x=time_idx_list, y=group_y, name="Group: {}".format(group_name),
                       line=dict(color='royalblue', width=1))
        )

        return fig

    def _add_group_step_decision_to_plot(self,
                                         fig,
                                         test_df,
                                         group_name,
                                         time_idx,
                                         reward,
                                         action,
                                         terminal,
                                         restart,
                                         steps_from_alert,
                                         restart_steps):
        y_value = list(
            test_df[(test_df[self.config.get("GroupKeyword")] == group_name) & (test_df.time_idx == time_idx)]
            [self.config.get("ValueKeyword")].values)

        fig.add_trace(
            go.Scatter(x=[time_idx],
                       y=y_value,
                       hovertext="StepsFromAlert: {}, \n"
                                 "RestartSteps: {}, \n"
                                 "Action: {}, \n"
                                 "Reward: {}, \n"
                                 "Terminal: {}, \n"
                                 "Restart: {}"
                                 "".format(steps_from_alert,
                                           restart_steps,
                                           action,
                                           reward,
                                           terminal,
                                           restart),
                       mode="markers",
                       showlegend=False,
                       marker=dict(
                           color="orange" if restart else
                           "purple" if terminal else
                           'green' if not action
                           else "red"
                       )
                       )
        )
        return fig

    def _get_min_test_time_idx(self, test_df):
        return test_df.time_idx.min() + self.config.get("EncoderLength") - 1
