import pandas as pd
from env_thts_common import get_num_iterations, get_group_idx_mapping, get_group_names, \
    get_group_lower_and_upper_bounds, is_group_prediction_out_of_bound, calc_reward, \
    is_state_terminal
import os
import numpy as np
import time
from Algorithms.render import render


class TrajectorySample:
    def __init__(self, env, config, deepar_model, val_df, test_df, num_trajectories=1000):
        assert os.getenv("MODEL_NAME") == "DeepAR", "Must use deep ar model to run TrajectorySample algorithm"

        self.env = env
        self.config = config
        self.enc_len = self.config.get("EncoderLength")
        self.pred_len = self.config.get("PredictionLength")
        self.max_steps_from_alert = config.get("Env").get("AlertMaxPredictionSteps") + 1
        self.max_restart_steps = config.get("Env").get("RestartSteps") + 1
        self.deepar_model = deepar_model
        self.val_df = val_df
        self.test_df = test_df
        self.num_trajectories = num_trajectories

        self.group_idx_mapping = get_group_idx_mapping(self.config, self.deepar_model, test_df)
        self.group_names = get_group_names(self.group_idx_mapping)

        self.actions = [0, 1]
        self.group_action_product_dict = self._define_actions()

        self.rollout_policy = self._define_rollout_policy()

    def run(self):
        prediction_df = pd.DataFrame()
        num_iterations = get_num_iterations(self.test_df, self.enc_len)

        action_history = []
        reward_history = []
        terminal_history = [{group_name: False for group_name in self.group_names}]
        restart_history = [{group_name: False for group_name in self.group_names}]
        alert_prediction_steps_history = [{group_name: self.max_steps_from_alert for group_name in self.group_names}]
        restart_steps_history = [{group_name: self.max_restart_steps for group_name in self.group_names}]

        statistics = self._init_statistics()
        for iteration in range(1, num_iterations):
            start = time.time()

            prediction_df = self._build_prediction_df(prediction_df, iteration)
            next_state_df = prediction_df[prediction_df.time_idx == (prediction_df.time_idx.max() - self.pred_len + 1)]
            prediction, x = self.deepar_model.predict(prediction_df,
                                                      mode="raw",
                                                      return_x=True,
                                                      n_samples=self.num_trajectories)
            action_group_mapping_dict = self._choose_action_per_group(prediction,
                                                                      terminal_history[-1],
                                                                      restart_history[-1],
                                                                      alert_prediction_steps_history[-1])
            reward_group_mapping, \
            next_state_group_terminal_mapping, \
            next_state_group_restart_mapping, \
            next_state_group_restart_steps_mapping, \
            next_state_group_steps_from_alert_mapping = self._transition_real_env(action_group_mapping_dict,
                                                                                  terminal_history[-1],
                                                                                  restart_history[-1],
                                                                                  alert_prediction_steps_history[-1],
                                                                                  restart_steps_history[-1],
                                                                                  next_state_df)

            statistics = self._update_statistics(statistics, reward_group_mapping, action_group_mapping_dict)

            action_history.append(action_group_mapping_dict)
            reward_history.append(reward_group_mapping)
            terminal_history.append(next_state_group_terminal_mapping)
            restart_history.append(next_state_group_restart_mapping)
            restart_steps_history.append(next_state_group_restart_steps_mapping)
            alert_prediction_steps_history.append(next_state_group_steps_from_alert_mapping)

            end = time.time()
            run_time = end - start

            render(self.config,
                   self.group_names,
                   self.test_df,
                   run_time,
                   action_history,
                   reward_history,
                   terminal_history,
                   restart_history,
                   alert_prediction_steps_history,
                   restart_steps_history)

        print(statistics)

    def _choose_action_per_group(self,
                                 prediction,
                                 terminal_mapping_previous_state,
                                 restart_mapping_previous_state,
                                 alert_prediction_steps_mapping_previous_state):
        if 'prediction' in prediction:
            prediction = prediction['prediction']

        feasible_alert_groups = self._get_feasible_alert_groups(terminal_mapping_previous_state,
                                                                restart_mapping_previous_state,
                                                                alert_prediction_steps_mapping_previous_state)
        scores = {}
        for group_name, actions in self.group_action_product_dict.items():
            if group_name in feasible_alert_groups:
                group_action_scores = []
                out_of_bound_list_tuple, _ = self._get_out_ob_bound_list_tuple(prediction, group_name)
                for action in actions:
                    score = 0
                    for out_of_bound, out_of_bound_idx in out_of_bound_list_tuple:
                        reward = 0
                        if action == 0 and out_of_bound:
                            reward = calc_reward(self.config,
                                                 good_alert=False,
                                                 false_alert=False,
                                                 missed_alert=True,
                                                 current_steps_from_alert=self.max_steps_from_alert - out_of_bound_idx,
                                                 max_steps_from_alert=self.max_steps_from_alert)
                        elif action == 1 and out_of_bound:
                            reward = calc_reward(self.config,
                                                 good_alert=True,
                                                 false_alert=False,
                                                 missed_alert=False,
                                                 current_steps_from_alert=self.max_steps_from_alert - out_of_bound_idx,
                                                 max_steps_from_alert=self.max_steps_from_alert)
                        elif action == 1 and not out_of_bound:
                            reward = calc_reward(self.config,
                                                 good_alert=False,
                                                 false_alert=True,
                                                 missed_alert=False,
                                                 current_steps_from_alert=self.max_steps_from_alert - out_of_bound_idx,
                                                 max_steps_from_alert=self.max_steps_from_alert)
                        score += reward
                    group_action_scores.append(score)
            else:
                group_action_scores = [1, -1]
            scores[group_name] = group_action_scores

        chosen_group_action_mapping = {
            group_name: np.argmax(group_action_scores)
            for group_name, group_action_scores
            in scores.items()
        }
        return chosen_group_action_mapping

    def _get_feasible_alert_groups(self,
                                   previous_state_terminal_dict,
                                   previous_state_restart_dict,
                                   previous_state_alert_prediction_steps_dict):
        feasible_alert_groups = []
        for group_name in self.group_names:
            if not previous_state_terminal_dict[group_name] \
                    and not previous_state_restart_dict[group_name] \
                    and previous_state_alert_prediction_steps_dict[group_name] == self.max_steps_from_alert:
                feasible_alert_groups.append(group_name)
        return feasible_alert_groups

    def _transition_real_env(self,
                             action_group_mapping_dict,
                             previous_state_terminal_dict,
                             previous_state_restart_dict,
                             previous_state_alert_prediction_steps_dict,
                             previous_state_restart_steps_dict,
                             next_state_df):

        reward_group_mapping = {}
        next_state_group_terminal_mapping = {}
        next_state_group_restart_mapping = {}
        next_state_group_restart_steps_mapping = {}
        next_state_group_steps_from_alert_mapping = {}

        for group_name in self.group_names:
            reward = 0
            action = action_group_mapping_dict[group_name]
            group_terminal = previous_state_terminal_dict[group_name]
            group_restart = previous_state_restart_dict[group_name]
            group_restart_steps = previous_state_restart_steps_dict[group_name]
            group_steps_from_alert = previous_state_alert_prediction_steps_dict[group_name]

            next_state_group_value = float(next_state_df[next_state_df[self.config.get("GroupKeyword")] == group_name][
                                               self.config.get("ValueKeyword")])
            group_next_state_terminal = is_state_terminal(self.config, group_name, next_state_group_value)
            next_state_group_terminal_mapping[group_name] = group_next_state_terminal

            if group_terminal or (group_restart and group_restart_steps > 1):
                next_state_group_terminal_mapping[group_name] = False
                next_state_group_restart_mapping[group_name] = True
                next_state_group_restart_steps_mapping[group_name] = group_restart_steps - 1
                next_state_group_steps_from_alert_mapping[group_name] = self.max_steps_from_alert

            elif (group_restart and group_restart_steps == 1) or group_steps_from_alert == 1:
                next_state_group_restart_mapping[group_name] = False
                next_state_group_restart_steps_mapping[group_name] = self.max_restart_steps
                next_state_group_steps_from_alert_mapping[group_name] = self.max_steps_from_alert

                if group_steps_from_alert == 1:
                    if not group_next_state_terminal:
                        reward += calc_reward(config=self.config,
                                              good_alert=False,
                                              false_alert=True,
                                              missed_alert=False,
                                              current_steps_from_alert=group_steps_from_alert,
                                              max_steps_from_alert=self.max_steps_from_alert)
                    else:
                        reward += calc_reward(config=self.config,
                                              good_alert=True,
                                              false_alert=False,
                                              missed_alert=False,
                                              current_steps_from_alert=group_steps_from_alert,
                                              max_steps_from_alert=self.max_steps_from_alert)

            elif 1 < group_steps_from_alert < self.max_steps_from_alert:
                next_state_group_restart_mapping[group_name] = False
                next_state_group_restart_steps_mapping[group_name] = self.max_restart_steps
                next_state_group_steps_from_alert_mapping[group_name] = group_steps_from_alert - 1

                if group_next_state_terminal:
                    reward += calc_reward(config=self.config,
                                          good_alert=True,
                                          false_alert=False,
                                          missed_alert=False,
                                          current_steps_from_alert=group_steps_from_alert,
                                          max_steps_from_alert=self.max_steps_from_alert)

            else:
                if action == 0:
                    next_state_group_restart_mapping[group_name] = False
                    next_state_group_restart_steps_mapping[group_name] = self.max_restart_steps
                    next_state_group_steps_from_alert_mapping[group_name] = self.max_steps_from_alert

                    if group_next_state_terminal:
                        reward += calc_reward(config=self.config,
                                              good_alert=False,
                                              false_alert=False,
                                              missed_alert=True,
                                              current_steps_from_alert=group_steps_from_alert,
                                              max_steps_from_alert=self.max_steps_from_alert)

                elif action == 1:
                    next_state_group_restart_mapping[group_name] = False
                    next_state_group_restart_steps_mapping[group_name] = self.max_restart_steps
                    next_state_group_steps_from_alert_mapping[group_name] = group_steps_from_alert - 1

                    if group_next_state_terminal:
                        reward += calc_reward(config=self.config,
                                              good_alert=True,
                                              false_alert=False,
                                              missed_alert=False,
                                              current_steps_from_alert=group_steps_from_alert,
                                              max_steps_from_alert=self.max_steps_from_alert)

            reward_group_mapping[group_name] = reward

        return reward_group_mapping, \
               next_state_group_terminal_mapping, \
               next_state_group_restart_mapping, \
               next_state_group_restart_steps_mapping, \
               next_state_group_steps_from_alert_mapping

    def _get_out_ob_bound_list_tuple(self, prediction, group_name):
        group_prediction_trajectories_samples = \
            prediction[self.group_idx_mapping[group_name]][:self.max_steps_from_alert].T
        lb = self.rollout_policy[group_name]['lb']
        ub = self.rollout_policy[group_name]['ub']
        out_of_bound_list_tuple = list(map(lambda x: is_group_prediction_out_of_bound(
            x, lb, ub), group_prediction_trajectories_samples))
        is_out_of_bound_list = [out_of_bound for out_of_bound, out_of_bound_idx in out_of_bound_list_tuple]
        out_of_bound_percent = sum(np.where(is_out_of_bound_list, 1, 0)) / float(len(out_of_bound_list_tuple))
        return out_of_bound_list_tuple, out_of_bound_percent

    def _build_prediction_df(self, prediction_df: pd.DataFrame, iteration):
        start_time_idx_test_df = self.val_df.time_idx.max() + 1
        end_time_idx_test_df = start_time_idx_test_df + iteration
        if prediction_df.empty:
            time_idx_range = list(range(start_time_idx_test_df, end_time_idx_test_df))
            prediction_df = pd.concat([self.val_df, self.test_df[self.test_df.time_idx.isin(time_idx_range)]], axis=0)
        else:
            start_time_idx_test_df = end_time_idx_test_df - 1
            time_idx_range = list(range(start_time_idx_test_df, end_time_idx_test_df))
            prediction_df = pd.concat([prediction_df, self.test_df[self.test_df.time_idx.isin(time_idx_range)]], axis=0)
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def _define_actions(self):
        actions_dict = {group_name: self.actions for group_name in self.group_names}
        return actions_dict

    def _define_rollout_policy(self):
        policy = {}
        for group_name in self.group_names:
            policy[group_name] = {}
            lb, ub = get_group_lower_and_upper_bounds(self.config, group_name)
            policy[group_name]['lb'] = lb + 0.3
            policy[group_name]['ub'] = ub - 0.3
        return policy

    def _init_statistics(self):
        statistics = {}
        for group_name in self.group_names:
            statistics[group_name] = {
                "TP": 0,
                "FN": 0,
                "FP": 0,
                "TN": 0
            }
        return statistics

    def _update_statistics(self, statistics, reward_group_mapping, action_group_mapping_dict):
        reward_false_alert = self.config.get("Env").get("Rewards").get("FalseAlert")
        reward_missed_alert = self.config.get("Env").get("Rewards").get("MissedAlert")

        for group_name in self.group_names:
            group_statistics = statistics[group_name]
            reward = reward_group_mapping[group_name]
            action = action_group_mapping_dict[group_name]

            if reward == 0:
                group_statistics["TN"] += 1
            elif reward > 0:
                group_statistics["TP"] += 1
            elif reward < 0:
                if reward_missed_alert != reward_false_alert:
                    if reward == reward_missed_alert:
                        group_statistics["FN"] += 1
                    elif reward == reward_false_alert:
                        group_statistics["FP"] += 1
                else:
                    if action == 0:
                        group_statistics["FN"] += 1
                    elif action == 1:
                        group_statistics["FP"] += 1
        return statistics