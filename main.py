import pandas as pd
from config import get_config
from Models.tft import create_tft_model, optimize_tft_hp
from Models.trainer import create_trainer, fit, get_model_from_trainer, get_model_from_checkpoint
from Models.deep_ar import create_deepar_model, optimize_deepar_hp
from evaluation import evaluate
import warnings
from data_utils import get_dataloader
import os
from utils import save_to_pickle, init_base_folders , load_pickle
from plot import plot_predictions, plot_data
from DataBuilders.fisherman import FishermanDataBuilder
import gym
from Algorithms.thts.max_uct import MaxUCT
from Algorithms.trajectory_sample.trajectory_sample import TrajectorySample

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def build_data(config):
    data_builder = FishermanDataBuilder(config)
    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds = data_builder.build_ts_data()

    train_dl = get_dataloader(train_ts_ds, is_train=True, config=config)
    val_dl = get_dataloader(val_ts_ds, is_train=False, config=config)
    test_dl = get_dataloader(test_ts_ds, is_train=False, config=config)

    save_to_pickle(val_df, config.get("val_df_pkl_path"))
    save_to_pickle(test_df, config.get("test_df_pkl_path"))

    return train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl


def optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl):
    model_name = config.get("model")
    if model_name not in ["TFT", "DeepAR"]:
        raise ValueError("model must be TFT or DeepAR")

    study = None
    is_study = config.get("study")
    study_path = config.get("load_study_path")
    if is_study or study_path:
        if study_path and os.path.isfile(study_path):
            study = load_pickle(study_path)
        else:
            if model_name == "TFT":
                study = optimize_tft_hp(config, train_dl, val_dl)

            elif model_name == "DeepAR":
                study = optimize_deepar_hp(config, train_dl, val_dl)

    is_train = config.get("train")
    model = None

    if model_name == "TFT":
        model = create_tft_model(train_ts_ds, study)
    elif model_name == "DeepAR":
        model = create_deepar_model(train_ts_ds, study)

    if study:
        gradient_clip_val = study.best_params['gradient_clip_val']
    else:
        gradient_clip_val = 0.01

    trainer = create_trainer(config, gradient_clip_val)
    if is_train:
        trainer = fit(trainer, model, train_dl, val_dl)
        fitted_model = get_model_from_trainer(trainer, model_name)
    else:
        fitted_model = get_model_from_checkpoint(config.get("load_model_path"), model_name)
    return fitted_model


if __name__ == '__main__':
    init_base_folders()
    config = get_config()

    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl \
        = build_data(config)

    if config.get("plot_data"):
        plot_data(config, pd.concat([train_df, val_df, test_df], axis=0))

    fitted_model = optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl)
    # evaluate(config, fitted_model, val_dl)

    if config.get("plot_predictions"):
        plot_predictions(config, fitted_model, val_dl, val_df)
        plot_predictions(config, fitted_model, test_dl, test_df)

    ad_env = gym.make("gym_ad:ad-v0")
    thts = MaxUCT(fitted_model, ad_env, config)
    thts.run(test_df)

    # trajectory_sample = TrajectorySample(ad_env, config, fitted_model, val_df, test_df, num_trajectories=5000)
    # trajectory_sample.run()
