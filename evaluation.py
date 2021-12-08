import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error, confusion_matrix, mean_absolute_error
from data_utils import get_dataloader, get_group_id_group_name_mapping
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
    print("MAE: {} MSE: {}".format(mae, mse))


def evaluate_regression_groups(config, ts_ds, actual, predictions):
    ts_ds_index = ts_ds.index.reset_index()
    group_id_group_name_mapping = get_group_id_group_name_mapping(config, ts_ds)

    for group_id, group_name in group_id_group_name_mapping.items():
        group_indices = ts_ds_index[ts_ds_index["group_id"] == group_id].index
        evaluate_regression_group(actual, predictions, group_name, group_indices)


def evaluate_regression_group(actual, predictions, group_name, group_indices):
    group_mse = mean_squared_error(actual[group_indices], predictions[group_indices])
    group_mae = mean_absolute_error(actual[group_indices], predictions[group_indices])
    print("Group: {}, MAE: {}, MSE: {}".format(group_name, group_mae, group_mse))


def evaluate_classification(config, ts_ds, model):
    dl = get_dataloader(ts_ds, False, config)
    actual = get_actuals(dl)
    predictions, x = model.predict(dl, return_x=True, show_progress_bar=True)

    get_classification_evaluation_summary(actual, predictions)


def get_classification_evaluation_summary(actual, predictions):
    tn, fp, fn, tp = confusion_matrix(actual.reshape(-1, 1), predictions.reshape(-1, 1)).ravel()
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