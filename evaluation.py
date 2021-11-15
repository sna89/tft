import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error, confusion_matrix, mean_absolute_error
from data_utils import get_dataloader, get_series_name_idx_mapping
import pandas as pd
import os

# with torch.no_grad():
    #     for x, _ in dl:
    #         ppp = model.forward(x, True)


def get_actuals(dl):
    actuals = torch.cat([y[0] for x, y in iter(dl)])
    return actuals


def evaluate_regression(config, ts_ds, model):
    dl = get_dataloader(ts_ds, False, config)
    actuals = get_actuals(dl)
    predictions, x = model.predict(dl, mode="prediction", return_x=True, show_progress_bar=True)
    print()
    print()

    evaluate_regression_groups(config, ts_ds, actuals, predictions)

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print("MSE: {}, MAE: {}".format(mse, mae))


def evaluate_regression_groups(config, ts_ds, actual, predictions):
    ts_ds_index = ts_ds.index.reset_index()
    group_idx_mapping = get_series_name_idx_mapping(config, ts_ds)

    for group_id, group_name in group_idx_mapping.items():
        group_indices = ts_ds_index[ts_ds_index["group_id"] == group_id].index
        group_name = ts_ds.decoded_index.iloc[group_indices.min()][config.get("GroupKeyword")]
        evaluate_regression_group(actual, predictions, group_name, group_indices)


def evaluate_regression_group(actual, predictions, group_name, group_indices):
    group_mse = mean_squared_error(actual[group_indices], predictions[group_indices])
    group_mae = mean_absolute_error(actual[group_indices], predictions[group_indices])
    print("Group: {}, MSE: {}, MAE: {}".format(group_name, group_mse, group_mae))


def evaluate_classification(config, ts_ds, model):
    dl = get_dataloader(ts_ds, False, config)
    actuals = get_actuals(dl)
    predictions, x = model.predict(dl, return_x=True, show_progress_bar=True)

    tn, fp, fn, tp = confusion_matrix(actuals.reshape(-1, 1), predictions.reshape(-1, 1)).ravel()
    print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))

    precision = tp / (float(tp + fp))
    print("Precision: {}".format(precision))

    recall = tp / (float(tp + fn))
    print("Recall: {}".format(recall))

    f_score = 2 * (precision * recall) / (precision + recall)
    print("F-Score: {}".format(f_score))


def evaluate_base_model(dl):
    actuals = torch.cat([y for x, (y, weight) in iter(dl)])
    baseline_predictions = Baseline().predict(dl)
    mse = mean_squared_error(actuals, baseline_predictions)
    mae = mean_absolute_error(actuals, baseline_predictions)
    return mse, mae