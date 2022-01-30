from Algorithms.statistics import init_statistics, update_statistics, output_statistics
from DataBuilders.build import convert_df_to_ts_data
from EnvCommon.env_thts_common import calc_reward
from config import REGRESSION_TASK_TYPE, ROLLOUT_TASK_TYPE, get_env_steps_from_alert, get_restart_steps
import os
from data_utils import get_dataloader, get_group_lower_and_upper_bounds, get_idx_list
from evaluation import evaluate_classification, get_actual_list, trim_last_samples, \
    get_classification_evaluation_summary
from utils import get_model_from_checkpoint
from Tasks.time_series_task import get_model_name
import pandas as pd
import numpy as np
import torch


def get_reg_fitted_model():
    model_name = get_model_name(REGRESSION_TASK_TYPE)
    checkpoint = os.getenv("CHECKPOINT_{}".format(REGRESSION_TASK_TYPE.upper()))
    reg_fitted_model = get_model_from_checkpoint(checkpoint, model_name)
    return reg_fitted_model


def run_rollout_task(config,
                     dataset_name,
                     train_df,
                     test_df):
    restart_steps = get_restart_steps(config)
    prediction_steps = get_env_steps_from_alert(config)

    reward_mapping = {}

    train_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, ROLLOUT_TASK_TYPE)
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, ROLLOUT_TASK_TYPE)
    test_dataloader = get_dataloader(test_ts_ds, False, config)

    actual_list = get_actual_list(test_dataloader, num_targets=1)
    actual = actual_list[0]

    reg_fitted_model = get_reg_fitted_model()
    predictions, x = reg_fitted_model.predict(test_dataloader, mode="prediction", return_x=True, show_progress_bar=True)

    group_names = x['groups'].unique()
    statistics = init_statistics(group_names)

    for group_name in group_names:
        group_idx_list = torch.nonzero(x['groups'] == group_name).T[0]
        reward = 0
        restart_counter = 0

        for idx in group_idx_list:
            prediction = predictions[idx][:prediction_steps]
            y = actual[idx][:prediction_steps]

            current_steps_from_alert = prediction_steps
            if restart_counter > 0:
                restart_counter -= 1
                statistics = update_statistics(config,
                                               group_name,
                                               statistics,
                                               0,
                                               current_steps_from_alert)
                if restart_counter == 1:
                    if not ((lb <= y[0]) & (y[0] <= ub)):
                        restart_counter = restart_steps

            missed_alert = False
            good_alert = False
            false_alert = False

            group_name = x['groups'][idx].item()
            lb, ub = get_group_lower_and_upper_bounds(config, str(group_name))

            rollout_actual = np.where((lb <= y) & (y <= ub), 0, 1)

            rollout_decision = np.where((lb <= prediction) & (prediction <= ub), 0, 1)
            rollout_decision = min(1, int(np.sum(rollout_decision)))

            if rollout_decision == 0:
                if not ((lb <= y[0]) & (y[0] <= ub)):
                    missed_alert = True
                    restart_counter = restart_steps

            elif rollout_decision == 1:
                if np.sum(rollout_actual) >= 1:
                    good_alert = True
                    current_steps_from_alert = prediction_steps - np.argmax(rollout_actual)
                    restart_counter = restart_steps + np.argmax(rollout_actual)

                else:
                    false_alert = True
                    current_steps_from_alert = 1
                    restart_counter = prediction_steps

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

        reward_mapping[group_name] = reward

    output_statistics(statistics, reward_mapping, filter_group_name=None)
