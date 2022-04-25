import torch
from pytorch_forecasting import Baseline
from sklearn.metrics import mean_squared_error, confusion_matrix, mean_absolute_error
from Utils.data_utils import get_dataloader, get_group_id_group_name_mapping


def get_actual_and_predictions(config, ts_ds, model, num_targets):
    dl = get_dataloader(ts_ds, False, config)

    actual_list = get_actual_list(dl, num_targets)
    predictions, x = model.predict(dl, mode="prediction", return_x=True, show_progress_bar=True)

    return actual_list, predictions, x


def get_actual_list(dl, num_targets):
    if num_targets == 1:
        ly = [y[0] for x, y in iter(dl)]
        actual = torch.cat(ly)
        return [actual]
    else:
        actual_list = []
        for num_target in range(num_targets):
            ly = [y[0][num_target] for x, y in iter(dl)]
            actual = torch.cat(ly)
            actual_list.append(actual)
        return actual_list


def evaluate_regression(config, ts_ds, model, num_targets):
    actual_list, predictions, x = get_actual_and_predictions(config, ts_ds, model, num_targets)
    print()

    for num_target in range(num_targets):
        target = ts_ds.target_names[num_target]
        print(target)

        actual = actual_list[num_target]
        evaluate_regression_groups(config, ts_ds, actual, predictions)

        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        print("MAE: {} MSE: {}".format(mae, mse))


def evaluate_regression_groups(config, ts_ds, actual, predictions):
    ts_ds_index = ts_ds.index.reset_index()
    group_id_group_name_mapping = get_group_id_group_name_mapping(config, ts_ds)

    for group_id, group_name in group_id_group_name_mapping.items():
        group_indices = ts_ds_index[ts_ds_index["group_id"] == group_id].index
        evaluate_regression_group(actual, predictions, group_name, group_indices)


def evaluate_regression_group(actual, predictions, group_name, group_indices):
    group_mse = mean_squared_error(actual[group_indices], predictions[group_indices])
    group_rmse = mean_squared_error(actual[group_indices], predictions[group_indices], squared=False)
    group_mae = mean_absolute_error(actual[group_indices], predictions[group_indices])
    print("Group: {}, MAE: {}, MSE: {}, RMSE: {}".format(group_name, group_mae, group_mse, group_rmse))


def evaluate_classification(config, ts_ds, model, num_targets):
    actual_list, predictions, x = get_actual_and_predictions(config, ts_ds, model, num_targets)

    prediction_steps = config.get("Env").get("AlertMaxPredictionSteps")
    actual = trim_last_samples(actual_list, prediction_steps)
    predictions = trim_last_samples(predictions, prediction_steps)

    get_classification_evaluation_summary(actual, predictions)


def trim_last_samples(sample_list, k):
    if isinstance(sample_list, list):
        sample_list = sample_list[0]
    return torch.Tensor([min(sum(a[:k]), 1) for a in sample_list])


def get_classification_evaluation_summary(actual, predictions):
    tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
    print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))

    evaluate_classification_from_conf_matrix(tn, fp, fn, tp)


def evaluate_classification_from_conf_matrix(tn, fp, fn, tp):
    precision = tp / (float(tp + fp)) if tp + fp > 0 else 0
    print("Precision: {}".format(precision))

    recall = tp / (float(tp + fn)) if tp + fn > 0 else 0
    print("Recall: {}".format(recall))

    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print("F-Score: {}".format(f_score))

    return precision, recall, f_score


def evaluate_base_model(dl):
    actuals = torch.cat([y for x, (y, weight) in iter(dl)])
    baseline_predictions = Baseline().predict(dl)
    mse = mean_squared_error(actuals, baseline_predictions)
    mae = mean_absolute_error(actuals, baseline_predictions)
    return mse, mae


def evaluate_combined(config, ts_ds, model, num_targets):
    dl = get_dataloader(ts_ds, False, config)
    actual_list = get_actual_list(dl, num_targets)
    predictions, x = model.predict(dl, mode="prediction", return_x=True, show_progress_bar=True)

    evaluate_regression_groups(config, ts_ds, actual_list[0], predictions[0])
    mse = mean_squared_error(actual_list[0], predictions[0])
    mae = mean_absolute_error(actual_list[0], predictions[0])
    print("MAE: {} MSE: {}".format(mae, mse))

    prediction_steps = config.get("Env").get("AlertMaxPredictionSteps")
    actual = trim_last_samples(actual_list[1], prediction_steps)
    prediction = trim_last_samples(predictions[1], prediction_steps)

    get_classification_evaluation_summary(actual, prediction)
