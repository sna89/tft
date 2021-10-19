import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error, confusion_matrix, mean_absolute_error
from data_utils import get_dataloader, get_group_idx_mapping

# with torch.no_grad():
    #     for x, _ in dl:
    #         ppp = model.forward(x, True)


def get_actuals(dl):
    actuals = torch.cat([y[0] for x, y in iter(dl)])
    return actuals


def evaluate_regression(config, df, ts_ds, model):
    dl = get_dataloader(ts_ds, False, config)
    actuals = get_actuals(dl)
    predictions, x = model.predict(dl, mode="prediction", return_x=True, show_progress_bar=True)
    print()

    group_idx_mapping = get_group_idx_mapping(config, model, df)
    for group, idx in group_idx_mapping.items():
        mask = (x['groups'] == idx)
        non_zero_indices = torch.nonzero(mask)[:, 0]
        if len(non_zero_indices) > 0:
            group_mse = mean_squared_error(actuals[non_zero_indices], predictions[non_zero_indices])
            group_mae = mean_absolute_error(actuals[non_zero_indices], predictions[non_zero_indices])
            print("Group: {}, MSE: {}, MAE: {}".format(group, group_mse, group_mae))
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print("MSE: {}, MAE: {}".format(mse, mae))


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