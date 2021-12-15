from DataBuilders.build import convert_df_to_ts_data
from config import REGRESSION_TASK_TYPE, ROLLOUT_TASK_TYPE
import os
from data_utils import get_dataloader, get_group_lower_and_upper_bounds, get_idx_list
from evaluation import evaluate_classification, get_actual_list, trim_last_samples, \
    get_classification_evaluation_summary
from utils import get_model_from_checkpoint
from Tasks.time_series_task import get_model_name
import pandas as pd
import numpy as np


def get_reg_fitted_model():
    model_name = get_model_name(REGRESSION_TASK_TYPE)
    checkpoint = os.getenv("CHECKPOINT_{}".format(REGRESSION_TASK_TYPE.upper()))
    reg_fitted_model = get_model_from_checkpoint(checkpoint, model_name)
    return reg_fitted_model


def execute_rollout(reg_fitted_model, test_dataloader):
    test_predictions, x = reg_fitted_model.predict(test_dataloader, mode="prediction", return_x=True,
                                                   show_progress_bar=True)
    return test_predictions, x


def run_rollout_task(config,
                     dataset_name,
                     train_df,
                     test_df):
    rollout_predictions = []
    rollout_actual = []

    prediction_steps = config.get("Env").get("AlertMaxPredictionSteps")

    reg_fitted_model = get_reg_fitted_model()

    train_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, ROLLOUT_TASK_TYPE)
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, ROLLOUT_TASK_TYPE)
    test_dataloader = get_dataloader(test_ts_ds, False, config)

    actual_list = get_actual_list(test_dataloader, num_targets=1)
    actual = actual_list[0]

    predictions, x = execute_rollout(reg_fitted_model, test_dataloader)

    idx_list = get_idx_list(test_dataloader, x, step=1)
    for idx in idx_list:
        prediction = predictions[idx]
        y = actual[idx]

        group_name = x['groups'][idx].item()
        lb, ub = get_group_lower_and_upper_bounds(config, str(group_name), is_observed=True)

        rollout_prediction = np.where((lb <= prediction) & (prediction <= ub), 0, 1)
        rollout_predictions.append(rollout_prediction)

        rollout_y = np.where((lb <= y) & (y <= ub), 0, 1)
        rollout_actual.append(rollout_y)

    predictions = trim_last_samples(rollout_predictions, prediction_steps)
    actual = trim_last_samples(rollout_actual, prediction_steps)

    get_classification_evaluation_summary(actual, predictions)
