from datetime import datetime
import datetime

from Algorithms.render import render
from Algorithms.statistics import init_statistics, update_statistics, output_statistics
from DataBuilders.build import convert_df_to_ts_data
from EnvCommon.env_thts_common import calc_reward, is_state_restart, update_steps_from_alert, update_restart_steps, \
    get_reward_type_for_group, is_out_of_bounds
from config import REGRESSION_TASK_TYPE, ROLLOUT_TASK_TYPE, get_env_steps_from_alert, get_restart_steps, \
    DATETIME_COLUMN, CLASSIFICATION_TASK_TYPE
import os
from Utils.data_utils import get_dataloader, get_group_lower_and_upper_bounds, get_group_id_group_name_mapping
from evaluation import get_actual_list
from Utils.utils import get_model_from_checkpoint
from Tasks.time_series_task import get_model_name
import numpy as np
import torch
from EnvCommon.env_thts_common import get_last_val_time_idx
import pandas as pd


def add_dummy_data_to_df(config,
                         df):
    dummy_data_list = []

    last_time_idx = df['time_idx'].max()
    last_date = df[DATETIME_COLUMN].max() if DATETIME_COLUMN in df.columns else None

    dummy_data_sample = df[lambda x: x.time_idx == last_time_idx].to_dict('records')

    for decoder_step in range(config.get("PredictionLength") - 3):
        idx_diff = 1 + decoder_step
        for sample in dummy_data_sample:
            group = sample[config.get("GroupKeyword")]
            value = sample[config.get("ValueKeyword")]
            data = {config.get("GroupKeyword"): group,
                    config.get("ValueKeyword"): value,
                    DATETIME_COLUMN: last_date + datetime.timedelta(
                        hours=idx_diff) if last_date else None,
                    'time_idx': last_time_idx + idx_diff
                    }
            dummy_data_list.append(data)

    dummy_df = pd.DataFrame.from_dict(dummy_data_list)
    df = pd.concat([df, dummy_df], axis=0).reset_index(drop=True)
    return df


def choose_action(prediction, lb, ub, task_type):
    rollout_decision = None

    if task_type == REGRESSION_TASK_TYPE:
        rollout_decision = np.where((lb <= prediction) & (prediction <= ub), 0, 1)
    elif task_type == CLASSIFICATION_TASK_TYPE:
        rollout_decision = prediction

    if isinstance(rollout_decision, np.ndarray):
        rollout_decision = min(1, int(np.sum(rollout_decision)))
    elif torch.is_tensor(rollout_decision):
        rollout_decision = min(1, int(torch.sum(rollout_decision)))
    else:
        raise ValueError

    return rollout_decision


def _is_rollout_terminal(y, lb, ub):
    rollout_actual = np.where((lb <= y) & (y <= ub), 0, 1)
    return np.sum(rollout_actual) >= 1


def _is_next_state_terminal(y, lb, ub):
    return not ((lb <= y[0]) & (y[0] <= ub))


def _is_restart(current_restart_steps):
    return current_restart_steps > 0


def get_fitted_model(task_type):
    model_name = get_model_name(task_type)
    checkpoint = os.getenv("CHECKPOINT_{}".format(task_type.upper()))
    fitted_model = get_model_from_checkpoint(checkpoint, model_name)
    return fitted_model


def run_rollout_task(config,
                     dataset_name,
                     test_df,
                     task_type=REGRESSION_TASK_TYPE):

    env_restart_steps = get_restart_steps(config)
    env_steps_from_alert = get_env_steps_from_alert(config)
    prediction_steps = get_env_steps_from_alert(config)

    val_last_time_idx = get_last_val_time_idx(config, test_df)
    test_last_time_idx = test_df["time_idx"].max()
    test_df = add_dummy_data_to_df(config, test_df)

    reward_mapping = {}
    action_history_mapping = {}
    reward_history_mapping = {}
    terminal_history_mapping = {}
    restart_history_mapping = {}
    steps_from_alert_history_mapping = {}
    restart_steps_history_mapping = {}

    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, None, ROLLOUT_TASK_TYPE)
    test_dataloader = get_dataloader(test_ts_ds, False, config)

    actual_list = get_actual_list(test_dataloader, num_targets=1)
    actual = actual_list[0]

    fitted_model = get_fitted_model(task_type)
    predictions, x = fitted_model.predict(test_dataloader, mode="prediction", return_x=True, show_progress_bar=True)

    group_id_group_name_mapping = get_group_id_group_name_mapping(config, test_ts_ds)
    group_names = list(group_id_group_name_mapping.values())
    statistics = init_statistics(group_names)

    for i, group_id_tensor in enumerate(x['groups'].unique()):
        current_steps_from_alert = env_steps_from_alert
        next_steps_from_alert = current_steps_from_alert
        current_restart_steps = 0

        is_current_normal = True
        is_current_terminal = False
        is_current_restart = False

        group_id = group_id_tensor.item()
        group_name = group_names[i]
        lb, ub = get_group_lower_and_upper_bounds(config, str(group_name))

        group_idx_list = torch.nonzero(x['groups'] == group_id).T[0]

        reward = 0
        action_history = []
        reward_history = []
        terminal_history = []
        restart_history = [False]
        steps_from_alert_history = []
        restart_steps_history = []

        for iteration, idx in enumerate(group_idx_list):
            if x['decoder_time_idx'][idx][0] <= val_last_time_idx:
                continue

            prediction = predictions[idx][:prediction_steps]
            next_actual = actual[idx][:prediction_steps]
            current_value = actual[idx - 1][0]

            good_alert = False
            false_alert = False
            missed_alert = False

            is_next_terminal = False
            is_next_restart = False
            is_next_normal = True

            if is_first_test_iteration(x, idx, val_last_time_idx):
                is_current_terminal = is_out_of_bounds(config, group_name, current_value)
                is_current_normal = not is_current_terminal
                terminal_history = [is_current_terminal]

            action = 0
            if is_current_normal:
                action = choose_action(prediction, lb, ub, task_type)
                is_next_terminal = is_out_of_bounds(config, group_name, next_actual[0])
                is_next_normal = not is_next_terminal
                if action == 0:
                    if is_next_terminal:
                        missed_alert = True
                else:
                    if is_next_terminal:
                        good_alert = True
                    else:
                        if env_steps_from_alert > 1:
                            next_steps_from_alert = current_steps_from_alert - 1
                            is_next_normal = False

                if current_restart_steps == 1:
                    current_restart_steps = 0

                if current_steps_from_alert == 1 and action == 0:
                    next_steps_from_alert = env_steps_from_alert

            else:
                is_current_alert = is_current_state_in_alert(current_steps_from_alert, env_steps_from_alert, action)

                if is_current_terminal:
                    next_steps_from_alert = env_steps_from_alert
                    current_restart_steps = env_restart_steps
                    is_next_terminal = False
                    is_next_restart = True
                    is_next_normal = False

                elif is_current_restart:
                    current_restart_steps -= 1

                    if current_restart_steps > 1:
                        is_next_restart = True
                        is_next_normal = False

                    elif current_restart_steps == 1:
                        is_next_restart = False
                        is_next_terminal = is_out_of_bounds(config, group_name, next_actual[0])
                        is_next_normal = not is_next_terminal

                elif is_current_alert:
                    is_next_terminal = is_out_of_bounds(config, group_name, next_actual[0])
                    is_next_normal = False

                    if is_next_terminal:
                        good_alert = True
                        next_steps_from_alert = env_steps_from_alert
                    else:
                        if current_steps_from_alert == 1:
                            false_alert = True
                            is_next_normal = True
                            next_steps_from_alert = env_steps_from_alert

                        else:
                            next_steps_from_alert = current_steps_from_alert - 1

            curr_reward = calc_reward(config,
                                      good_alert,
                                      false_alert,
                                      missed_alert,
                                      current_steps_from_alert,
                                      prediction_steps)

            statistics = update_statistics(config,
                                           group_name,
                                           statistics,
                                           curr_reward,
                                           current_steps_from_alert)

            reward += curr_reward

            action_history.append(action)
            reward_history.append(curr_reward)
            terminal_history.append(is_next_terminal)
            restart_history.append(is_next_restart)
            steps_from_alert_history.append(current_steps_from_alert)
            restart_steps_history.append(current_restart_steps)

            is_current_terminal = is_next_terminal
            is_current_restart = is_next_restart
            is_current_normal = is_next_normal
            current_steps_from_alert = next_steps_from_alert

        reward_mapping[group_name] = reward
        action_history_mapping[group_name] = action_history
        reward_history_mapping[group_name] = reward_history
        terminal_history_mapping[group_name] = terminal_history
        restart_history_mapping[group_name] = restart_history
        steps_from_alert_history_mapping[group_name] = steps_from_alert_history
        restart_steps_history_mapping[group_name] = restart_steps_history

    output_statistics(statistics, reward_mapping, None, task_type)

    for group_name in group_names:
        render(config,
               group_name,
               test_df[test_df["time_idx"] <= test_last_time_idx],
               action_history_mapping[group_name],
               reward_history_mapping[group_name],
               terminal_history_mapping[group_name],
               restart_history_mapping[group_name],
               steps_from_alert_history_mapping[group_name],
               restart_steps_history_mapping[group_name],
               task_type)


def is_first_test_iteration(x, idx, val_last_time_idx):
    return x['decoder_time_idx'][idx][0] == val_last_time_idx + 1


def is_current_state_in_alert(current_steps_from_alert, env_steps_from_alert, action):
    return (1 <= current_steps_from_alert < env_steps_from_alert) or (env_steps_from_alert == 1 and action == 1)