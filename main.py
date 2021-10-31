import pandas as pd
from config import get_config
from Models.trainer import optimize_hp, fit_regression_model, create_classification_model, fit_classification_model
from DataBuilders.build import build_data, split_df, convert_df_to_ts_data, process_data
from evaluation import evaluate_regression, evaluate_classification
import warnings
import os
from utils import save_to_pickle
from plot import plot_reg_predictions, plot_data
import gym
from Algorithms.thts.max_uct import MaxUCT
from Algorithms.trajectory_sample.trajectory_sample import TrajectorySample
from data_utils import get_label_weights
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from data_utils import get_dataloader


if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    model_name = os.getenv("MODEL_NAME")

    config = get_config(dataset_name)

    data = build_data(config, dataset_name)
    data = process_data(config, dataset_name, data)
    # plot_data(config, dataset_name, data)

    train_df, val_df, test_df = split_df(config, dataset_name, data)
    train_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, "reg")
    val_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, val_df, parameters, "reg")
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, "reg")

    reg_study = optimize_hp(config, train_ts_ds, val_ts_ds, model_name, type_="reg")
    fitted_reg_model = fit_regression_model(config, train_ts_ds, val_ts_ds, model_name, reg_study, type_="reg")

    evaluate_regression(config, test_ts_ds, fitted_reg_model)
    plot_reg_predictions(config, fitted_reg_model, test_df, test_ts_ds, dataset_name, model_name)

    save_to_pickle(val_df, config.get("ValDataFramePicklePath"))
    save_to_pickle(test_df, config.get("TestDataFramePicklePath"))
    ad_env = gym.make("gym_ad:ad-v0")
    thts = MaxUCT(ad_env, config)
    thts.run(test_df)

    # train_exc_df, val_exc_df, test_exc_df = split_df(config, dataset_name, pd.concat([val_df, test_df], axis=0))
    # train_exc_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, "class")
    # val_exc_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, val_df, parameters, "class")
    # test_exc_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, "class")

    # weights = get_label_weights(pd.concat([train_exc_df, val_exc_df], axis=0),
    #                             labels=[str(0), str(1)],
    #                             label_keyword=config.get("ExceptionKeyword"))
    # classification_model = create_classification_model(reg_study, weights)
    # fitted_classification_model = fit_classification_model(config, classification_model, train_exc_ts_ds, val_exc_ts_ds)
    # evaluate_classification(config, test_exc_ts_ds, fitted_classification_model)

    # trajectory_sample = TrajectorySample(ad_env, config, fitted_reg_model, val_df, test_df, num_trajectories=5000)
    # trajectory_sample.run()
