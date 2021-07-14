from thts.node import DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state
from env_thts_common import get_reward_and_terminal, build_next_state, EnvState, State
import time
import plotly.graph_objects as go


class TrialBasedHeuristicTree:
    def __init__(self, env, config):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.config = config

        self.env_name = config.get("env_name")
        self.num_trials = config.get("THTS").get("NumTrials")
        self.trial_length = config.get("THTS").get("TrialLength")
        self.uct_bias = config.get("THTS").get("UCTBias")
        self.runs = config.get("THTS").get("Runs")
        self.alert_prediction_steps = self.env.max_steps_from_alert
        self.consider_trial_length = True
        self.restart_env_iterations = 4

    def get_initial_state(self):
        initial_state = self.env.reset()
        return initial_state

    def run(self, test_df):
        state = self.get_initial_state()
        initial_node = DecisionNode(state, parent=None)

        for run in range(1, self.runs + 1):
            last_action = 0
            env_terminal = False
            restart_env_iterations = self.restart_env_iterations

            current_node = deepcopy(initial_node)

            action_history = []
            reward_history = []
            terminal_history = []
            alert_prediction_steps_history = []

            num_iterations = self._get_num_iterations(test_df)
            for iteration in range(1, num_iterations):
                action, \
                run_time, \
                restart_env_iterations, \
                    = self._before_transition(current_node,
                                              last_action,
                                              env_terminal,
                                              restart_env_iterations)

                next_state, reward, env_terminal = self._transition_real_env(current_node, test_df, iteration, action)

                next_state, reward, terminal, restart_env_iterations = \
                    self._after_transition(next_state,
                                           reward,
                                           env_terminal,
                                           restart_env_iterations)

                action_history.append(action)
                reward_history.append(reward)
                terminal_history.append(env_terminal)
                alert_prediction_steps_history.append(current_node.state.steps_from_alert)

                self.render(test_df,
                            run_time,
                            action_history,
                            reward_history,
                            terminal_history,
                            alert_prediction_steps_history)

                current_node = DecisionNode(next_state, parent=current_node, terminal=False)
                last_action = action

    def _before_transition(self, current_node, last_action, env_terminal, restart_env_iterations):
        action = 0
        run_time = 0
        if self.is_terminal(env_terminal, restart_env_iterations):
            restart_env_iterations -= 1
        elif last_action == 1 or (
                last_action == 0 and current_node.state.steps_from_alert < self.alert_prediction_steps):
            pass
        elif not self.is_terminal(env_terminal, restart_env_iterations) and last_action == 0:
            start = time.time()
            for trial in range(self.num_trials):
                self._run_trial(current_node)
            action = self.select_greedy_action(current_node)
            end = time.time()
            run_time = end - start
        return action, run_time, restart_env_iterations

    def _after_transition(self, next_state, reward, env_terminal, restart_env_iterations):
        terminal = False
        if self.is_terminal(env_terminal, restart_env_iterations):
            terminal = True
            next_state.steps_from_alert = self.env.max_steps_from_alert
            if restart_env_iterations < self.restart_env_iterations:
                reward = 0

            if env_terminal or restart_env_iterations == 0:
                restart_env_iterations = self.restart_env_iterations

        return next_state, reward, terminal, restart_env_iterations

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
        chance_node.visit()
        next_state, terminal, prob = self._select_outcome(chance_node)
        if self.consider_trial_length and depth == self.trial_length - 1:
            terminal = True

        # check if node already exists
        decision_node = self.add_decision_node(next_state, chance_node, terminal=terminal, prob=prob)
        self._visit_decision_node(decision_node, depth + 1, terminal)
        self._backup_chance_node(chance_node)

    def _select_outcome(self, chance_node: ChanceNode):
        set_env_to_state(self.env, chance_node.state)
        next_state, reward, terminal, prob = self.env.step(chance_node.action)  # monte carlo sample
        chance_node.reward = reward
        return next_state, terminal, prob

    def _initialize_decision_node(self, decision_node: DecisionNode):
        feasible_actions = self._get_feasible_actions_for_node(decision_node)
        for action in feasible_actions:
            self.add_chance_node(decision_node, action=action)

    def _get_feasible_actions_for_node(self, decision_node: DecisionNode):
        actions = list(range(self.env.action_space.n))
        steps_from_alert = decision_node.state.steps_from_alert
        if self.env.min_steps_from_alert < steps_from_alert < self.env.max_steps_from_alert:
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

        decision_node = TrialBasedHeuristicTree._get_decision_node_from_chance_node(chance_node, next_state)
        if not decision_node:
            decision_node = DecisionNode(state=next_state, parent=chance_node, terminal=terminal, prob=prob)
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

    def _transition_real_env(self, node: DecisionNode, test_df, iteration, action):
        val_max_time_idx = test_df.time_idx.min() + self.config.get("EncoderLength") - 1
        new_sample = test_df[lambda x: x.time_idx == (val_max_time_idx + iteration)]
        new_sample_values = list(new_sample['value'])

        next_state = build_next_state(node.state,
                                      new_sample_values,
                                      self.env.max_steps_from_alert,
                                      action)
        reward, terminal = get_reward_and_terminal(self.config,
                                                   list(new_sample['value']),
                                                   node.state.steps_from_alert,
                                                   action)
        return next_state, reward, terminal

    def render(self, test_df, run_time, action_history, reward_history, terminal_history,
               alert_prediction_steps_history):
        print("Action: {}".format(action_history[-1]))
        print("Reward: {}".format(reward_history[-1]))
        print("Terminal: {}".format(terminal_history[-1]))
        print("Iteration RunTime: {}".format(run_time))

        series_list = list(test_df.series.unique())
        min_time_idx = test_df.time_idx.min() + \
                       self.config.get("EncoderLength") - 1

        time_idx_list = list(test_df[test_df.time_idx >= min_time_idx]['time_idx'].unique())

        fig = go.Figure()
        for series in series_list:

            y = list(test_df[(test_df.series == series) & (test_df.time_idx.isin(time_idx_list))]['value'].values)
            fig.add_trace(
                go.Scatter(x=time_idx_list, y=y, name="series: {}".format(series),
                           line=dict(color='royalblue', width=1))
            )

            action_time_idx_list = list(time_idx_list[:len(action_history)])
            for idx, action_time_idx in enumerate(action_time_idx_list):
                action_y = list(
                    test_df[(test_df.series == series) & (test_df.time_idx == action_time_idx)]['value'].values)

                action = action_history[idx]
                reward = reward_history[idx]
                terminal = terminal_history[idx]
                steps_from_alert = alert_prediction_steps_history[idx]

                fig.add_trace(
                    go.Scatter(x=[action_time_idx],
                               y=action_y,
                               hovertext="StepsFromAlert: {},"
                                         "Action: {}, "
                                         "Reward: {}, "
                                         "Terminal: {}"
                                         "".format(steps_from_alert,
                                                   action,
                                                   reward,
                                                   terminal),
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   color="purple" if terminal else 'green' if not action else "red"
                               )
                               )
                )

        fig.update_xaxes(title_text="<b>time_idx</b>")
        fig.update_yaxes(title_text="<b>Actual</b>")
        fig.write_html('render_6.html')

    def _get_num_iterations(self, test_df):
        num_iterations = test_df.time_idx.max() - test_df.time_idx.min() - self.env.model_enc_len + 3
        return num_iterations
