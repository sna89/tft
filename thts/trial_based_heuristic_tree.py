from thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state
from env_thts_common import get_reward, build_next_state, EnvState, State, is_alertable_state, \
    get_series_lower_and_upper_bounds
import time
import plotly.graph_objects as go
import pandas as pd
from typing import List


class TrialBasedHeuristicTree:
    def __init__(self, env, config):
        self.env_name = "real"
        self.env = env
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
            terminal_history = []
            alert_prediction_steps_history = []

            num_iterations = self._get_num_iterations(test_df)
            for iteration in range(1, num_iterations):
                action, run_time = self._before_transition(current_node)
                action_list = self._postprocess_action(current_node, action, test_df, iteration)

                next_state, env_terminal_list, reward = self._transition_real_env(current_node,
                                                                                  test_df,
                                                                                  iteration,
                                                                                  action_list)

                next_state = self._after_transition(next_state, env_terminal_list)

                action_history.append(action_list)
                reward_history.append(reward)
                terminal_history.append(env_terminal_list)
                alert_prediction_steps_history.append([series_state.steps_from_alert
                                                       for series_state
                                                       in current_node.state.env_state])

                self.render(test_df,
                            run_time,
                            action_history,
                            reward_history,
                            terminal_history,
                            alert_prediction_steps_history)

                current_node = DecisionNode(next_state, parent=current_node, terminal=False)

    def _before_transition(self, current_node: DecisionNode):
        action = 0
        run_time = 0
        if not is_alertable_state(current_node, self.alert_prediction_steps, self.restart_env_iterations):
            pass
        else:
            start = time.time()
            for trial in range(self.num_trials):
                self._run_trial(current_node)
            action = self.select_greedy_action(current_node)
            end = time.time()
            run_time = end - start
        return action, run_time

    def _postprocess_action(self, current_node: DecisionNode, action: int, test_df: pd.DataFrame(), iteration: int) -> List[int]:
        if action == 0:
            return [action] * self.env.num_series
        else:
            action_list = []
            for series in range(self.env.num_series):
                restart_steps = current_node.state.env_state[series].restart_steps
                if restart_steps == self.env.max_restart_steps:
                    time_idx_range = self.get_volatility_time_idx_range(test_df, iteration)
                    if self.series_volatility_exceeds_bound(test_df, time_idx_range, series):
                        action_list.append(1)
                    else:
                        action_list.append(0)
                else:
                    action_list.append(0)
            return action_list

    def get_volatility_time_idx_range(self, test_df: pd.DataFrame(), iteration: int):
        min_time_idx = test_df.time_idx.min()
        time_idx_range_start = min_time_idx + iteration
        time_idx_range_end = time_idx_range_start + self.env.model_enc_len
        time_idx_range = list(range(time_idx_range_start, time_idx_range_end))
        return time_idx_range

    def series_volatility_exceeds_bound(self, test_df: pd.DataFrame, time_idx_range: List[int], series: int):
        series_df = test_df[(test_df.series == series) & (test_df.time_idx.isin(time_idx_range))]
        std = series_df.value.std()
        mean = series_df.value.mean()
        lb, ub = get_series_lower_and_upper_bounds(self.config, series)
        if (mean + 1.65 * std > ub) or (mean - 1.65 * std < lb):
            return True
        else:
            return False

    def _after_transition(self, next_state: EnvState, env_terminal_list: List[bool]):
        for series, series_state in enumerate(next_state.env_state):
            env_terminal = env_terminal_list[series]
            restart_steps = series_state.restart_steps
            if self.is_terminal(env_terminal, restart_steps):
                next_state.env_state[series].steps_from_alert = self.env.max_steps_from_alert

        return next_state

    def is_terminal(self, env_terminal, restart_env_iterations):
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

    def _transition_real_env(self, node: DecisionNode, test_df: pd.DataFrame(), iteration: int, action_list: List[int]):
        val_max_time_idx = test_df.time_idx.min() + self.config.get("EncoderLength") - 1
        new_sample = test_df[lambda x: x.time_idx == (val_max_time_idx + iteration)]
        new_sample_values = list(new_sample[self.config.get("ValueKeyword")])

        next_state, terminal_states = build_next_state(self.env_name,
                                                       self.config,
                                                       node.state,
                                                       new_sample_values,
                                                       self.env.max_steps_from_alert,
                                                       self.env.max_restart_steps,
                                                       action_list)

        reward = get_reward(self.env_name,
                            self.config,
                            list(new_sample[self.config.get("ValueKeyword")]),
                            node.state,
                            action_list)
        return next_state, terminal_states, reward

    def render(self,
               test_df: pd.DataFrame(),
               run_time: float,
               action_history: List[List[int]],
               reward_history: List[float],
               terminal_history: List[List[bool]],
               alert_prediction_steps_history: List[List[int]]):

        print("Action: {}".format(action_history[-1]))
        print("Reward: {}".format(reward_history[-1]))
        print("Iteration RunTime: {}".format(run_time))

        series_list = list(test_df.series.unique())

        min_time_idx = test_df.time_idx.min() + \
                       self.config.get("EncoderLength") - 1
        time_idx_list = list(test_df[test_df.time_idx >= min_time_idx]['time_idx'].unique())

        fig = go.Figure()
        for series in series_list:

            series_y = list(test_df[(test_df.series == series) & (test_df.time_idx.isin(time_idx_list))][self.config.get("ValueKeyword")].values)
            fig.add_trace(
                go.Scatter(x=time_idx_list, y=series_y, name="series: {}".format(series),
                           line=dict(color='royalblue', width=1))
            )

            current_time_idx_list = list(time_idx_list[:len(action_history)])
            for idx, time_idx in enumerate(current_time_idx_list):
                y_value = list(
                    test_df[(test_df.series == series) & (test_df.time_idx == time_idx)][self.config.get("ValueKeyword")].values)

                reward = reward_history[idx]
                action_per_series_list = action_history[idx]
                terminal_per_series_list = terminal_history[idx]
                steps_from_alert_per_series_list = alert_prediction_steps_history[idx]

                fig.add_trace(
                    go.Scatter(x=[time_idx],
                               y=y_value,
                               hovertext="StepsFromAlert: {},"
                                         "Action: {}, "
                                         "Reward: {}, "
                                         "Terminal: {}"
                                         "".format(steps_from_alert_per_series_list[series],
                                                   action_per_series_list[series],
                                                   reward,
                                                   terminal_per_series_list[series]),
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   color="purple"
                                   if terminal_per_series_list[series]
                                   else 'green' if not action_per_series_list[series]
                                   else "red"
                               )
                               )
                )

        fig.update_xaxes(title_text="<b>time_idx</b>")
        fig.update_yaxes(title_text="<b>Actual</b>")
        fig.write_html('render_6.html')

    def _get_num_iterations(self, test_df):
        num_iterations = test_df.time_idx.max() - test_df.time_idx.min() - self.env.model_enc_len + 3
        return num_iterations
