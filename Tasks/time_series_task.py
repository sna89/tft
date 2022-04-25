from DataBuilders.build import convert_df_to_ts_data
from Models.trainer import optimize_hp, fit_model
from Utils.data_utils import get_bound_col_name
from evaluation import evaluate_regression, evaluate_classification, evaluate_combined
from plot import plot_reg_predictions
from Loss.weighted_cross_entropy import WeightedCrossEntropy
from pytorch_forecasting import QuantileLoss, MultiLoss
from config import CLASSIFICATION_TASK_TYPE, REGRESSION_TASK_TYPE, COMBINED_TASK_TYPE, QUANTILES
import pandas as pd
import os


def run_time_series_task(config,
                         task_type,
                         dataset_name,
                         train_df,
                         val_df,
                         test_df,
                         evaluate=False,
                         plot=False):
    train_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, task_type)
    val_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, val_df, parameters, task_type)
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, None, task_type)

    model_name = get_model_name(task_type)
    weights = get_loss_weights(config, task_type, train_df, val_df)
    num_targets = get_num_targets(train_ts_ds)

    loss, output_size = get_loss(task_type,
                                 model_name,
                                 weights,
                                 num_targets)

    study = optimize_hp(config,
                        train_ts_ds,
                        val_ts_ds,
                        model_name,
                        task_type,
                        loss)

    fitted_model = fit_model(config,
                             task_type,
                             train_ts_ds,
                             val_ts_ds,
                             model_name,
                             loss,
                             output_size,
                             study)

    if evaluate:
        if task_type == REGRESSION_TASK_TYPE:
            evaluate_regression(config, test_ts_ds, fitted_model, num_targets)
        elif task_type == CLASSIFICATION_TASK_TYPE:
            evaluate_classification(config, val_ts_ds, fitted_model, num_targets)
        elif task_type == COMBINED_TASK_TYPE:
            evaluate_combined(config, val_ts_ds, fitted_model, num_targets)

    if plot:
        if task_type == REGRESSION_TASK_TYPE:
            plot_reg_predictions(config, fitted_model, test_df, test_ts_ds, dataset_name, model_name)

    return fitted_model


def get_label_weights(df, labels, label_keyword):
    weights = []
    for label in labels:
        label_count = df[df[label_keyword] == label].shape[0]
        weights.append(label_count)
    max_weight = max(weights)
    weights = [0 if weight == 0 else max_weight / weight
               for weight
               in weights]
    return weights


def get_loss_weights(config, task_type, train_df, val_df):
    loss_weights = None
    if os.getenv("DATASET") == "Synthetic":
        if task_type in [CLASSIFICATION_TASK_TYPE, COMBINED_TASK_TYPE]:
            bound_col = get_bound_col_name(config)
            loss_weights = get_label_weights(pd.concat([train_df, val_df], axis=0),
                                             labels=[str(0), str(1)],
                                             label_keyword=bound_col)

    return loss_weights


def get_loss(task_type="reg", model_name="TFT", weights=None, num_targets=1):
    if task_type == REGRESSION_TASK_TYPE:
        loss, output_size = get_regression_loss()

    elif task_type == COMBINED_TASK_TYPE and model_name == "TFT":
        loss = MultiLoss([QuantileLoss(QUANTILES),
                          WeightedCrossEntropy(weights)], weights=None)
        output_size = [len(QUANTILES), 2]

    elif task_type == CLASSIFICATION_TASK_TYPE:
        if model_name == "Mlp":
            loss = WeightedCrossEntropy(weights)
            output_size = 2
        else:
            raise ValueError

    else:
        raise ValueError

    if task_type != COMBINED_TASK_TYPE and num_targets > 1:
        output_size = [output_size] * num_targets

    return loss, output_size


def get_regression_loss():
    loss = QuantileLoss(QUANTILES)
    output_size = len(QUANTILES)
    return loss, output_size


def get_num_targets(ts_ds):
    return len(ts_ds.target_names)


def get_model_name(task_type):
    return os.getenv("MODEL_NAME_{}".format(task_type.upper()))
