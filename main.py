import pandas as pd
from config import get_config
from Models.trainer import optimize_hp, fit_regression
from DataBuilders.build import get_processed_data, split_df, convert_df_to_ts_data
from evaluation import evaluate_regression
import warnings
import os
from utils import save_to_pickle
from plot import plot_reg_predictions, plot_data
import gym
from Algorithms.thts.max_uct import MaxUCT
from Algorithms.trajectory_sample.trajectory_sample import TrajectorySample
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    model_name = os.getenv("MODEL_NAME")

    config = get_config(dataset_name)

    data = get_processed_data(config, dataset_name)
    # plot_data(config, dataset_name, data)

    train_df, val_df, test_df = split_df(config, dataset_name, data)
    train_ts_ds, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None)
    val_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, val_df, parameters)
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters)

    study = optimize_hp(config, train_ts_ds, val_ts_ds, model_name)
    fitted_reg_model = fit_regression(config, train_ts_ds, val_ts_ds, model_name, study)

    # evaluate_regression(config, test_df, test_ts_ds, fitted_reg_model)
    plot_reg_predictions(config, fitted_reg_model, test_df, test_ts_ds, dataset_name, model_name)

    # train_exp_df, val_exp_df, test_exp_df, train_exp_ts_ds, val_exp_ts_ds, test_exp_ts_ds, train_exp_dl, val_exp_dl, \
    #     test_exp_dl = build_exception_data(config, dataset_name, pd.concat([val_df, test_df], axis=1))
    # fitted_classification_model = fit_classification_model(config,  fitted_model, train_exp_dl, val_exp_dl)

    # save_to_pickle(val_df, config.get("ValDataFramePicklePath"))
    # save_to_pickle(test_df, config.get("TestDataFramePicklePath"))
    # ad_env = gym.make("gym_ad:ad-v0")
    # thts = MaxUCT(ad_env, config)
    # thts.run(test_df)
    # trajectory_sample = TrajectorySample(ad_env, config, fitted_model, val_df, test_df, num_trajectories=5000)
    # trajectory_sample.run()
