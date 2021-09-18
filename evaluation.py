import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error, confusion_matrix
from data_utils import get_group_idx_mapping


def evaluate(config, model, dl):
    actuals = torch.cat([y[0] for x, y in iter(dl)])
    predictions, x = model.predict(dl, return_x=True, show_progress_bar=True)
    print()
    group_idx_mapping = get_group_idx_mapping(config, model, dl)
    for group, idx in group_idx_mapping.items():
        mask = (x['groups'] == idx)
        non_zero_indices = torch.nonzero(mask)[:, 0]
        if len(non_zero_indices) > 0:
            group_mse = mean_squared_error(actuals[non_zero_indices], predictions[non_zero_indices])
            group_mae = calc_mae(actuals[non_zero_indices], predictions[non_zero_indices])
            print("Group: {}, MSE: {}, MAE: {}".format(group, group_mse, group_mae))
    mse = mean_squared_error(actuals, predictions)
    mae = calc_mae(actuals, predictions)
    print("MSE: {}, MAE: {}".format(mse, mae))
    return mse, mae


def evaluate_base_model(dl):
    actuals = torch.cat([y for x, (y, weight) in iter(dl)])
    baseline_predictions = Baseline().predict(dl)
    mse = mean_squared_error(actuals, baseline_predictions)
    mae = calc_mae(actuals, baseline_predictions)
    return mse, mae


def calc_mae(actuals, predictions):
    mae = (actuals - predictions).abs().mean()
    return mae