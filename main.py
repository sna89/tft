import pandas as pd
from plot import plot_predictions, plot_synthetic_data
from config import get_config
from Models.tft import create_trainer, create_tft_model, fit, get_model_from_trainer, get_model_from_checkpoint, \
    optimize_tft_hp
from evaluation import evaluate, evaluate_base_model
import warnings
from data_utils import get_dataloader
import os
from data_factory import get_data_builder
import gym
from utils import save_to_pickle
from thts.max_uct import MaxUCT
from thts.dp_uct import DpUCT


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def build_data(config, dataset_name):
    data_builder = get_data_builder(config, dataset_name)
    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds = data_builder.build_ts_data()
    train_dl = get_dataloader(train_ts_ds, is_train=True, config=config)
    val_dl = get_dataloader(val_ts_ds, is_train=False, config=config)
    test_dl = get_dataloader(test_ts_ds, is_train=False, config=config)
    return train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl


def optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl, to_fit=False):
    trainer = create_trainer()
    study = optimize_tft_hp(config, train_dl, val_dl)
    tft_model = create_tft_model(train_ts_ds, study)
    if to_fit:
        trainer = fit(trainer, tft_model, train_dl, val_dl)
        model = get_model_from_trainer(trainer)
    else:
        model = get_model_from_checkpoint(os.getenv("CHECKPOINT"))
    return model


if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    config = get_config(dataset_name)
    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl \
        = build_data(config, dataset_name)
    plot_synthetic_data(config, pd.concat([train_df, val_df, test_df], axis=0))
    save_to_pickle(val_df, config.get("ValDataFramePicklePath"))
    save_to_pickle(test_df, config.get("TestDataFramePicklePath"))
    fitted_model = optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl)
    # plot_predictions(fitted_model, test_dl, test_df, config, dataset_name)
    # evaluate(fitted_model, val_dl)
    tft_env = gym.make("gym_ad_tft:tft-v0")
    thts = DpUCT(tft_env, config)
    thts.run(test_df)
