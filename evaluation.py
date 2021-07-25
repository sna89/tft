import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error


def evaluate(model, dl, ad_env):
    actuals = torch.cat([y[0] for x, y in iter(dl)])
    predictions, x = model.predict(dl, return_x=True)
    group_idx_mapping = ad_env.group_idx_mapping
    for group, idx in group_idx_mapping.items():
        mask = (x['groups'] == idx)
        non_zero_indices = torch.nonzero(mask)[:, 0]
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