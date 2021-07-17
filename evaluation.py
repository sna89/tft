import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error


def evaluate(model, dl):
    actuals = torch.cat([y[0] for x, y in iter(dl)])
    predictions = model.predict(dl)
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