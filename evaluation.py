import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error


def evaluate(model, val_dl):
    actuals = torch.cat([y[0] for x, y in iter(val_dl)])
    predictions = model._predict_next_state(val_dl)
    mse = mean_squared_error(actuals, predictions)
    mae = calc_mae(actuals, predictions)
    print("MSE: {}, MAE: {}".format(mse, mae))
    return mse, mae


def evaluate_base_model(val_dl):
    actuals = torch.cat([y for x, (y, weight) in iter(val_dl)])
    baseline_predictions = Baseline().predict(val_dl)
    mse = mean_squared_error(actuals, baseline_predictions)
    mae = calc_mae(actuals, baseline_predictions)
    return mse, mae


def calc_mae(actuals, predictions):
    mae = (actuals - predictions).abs().mean()
    return mae