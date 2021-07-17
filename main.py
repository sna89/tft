import pandas as pd
from plot import plot_data, plot_predictions
from config import get_config
from Models.tft import create_tft_model, optimize_tft_hp
from Models.trainer import create_trainer, fit, get_model_from_trainer, get_model_from_checkpoint
from Models.deep_ar import create_deepar_model
from evaluation import evaluate, evaluate_base_model
import warnings
from data_utils import get_dataloader
import os
from data_factory import get_data_builder
import gym
from utils import save_to_pickle
from thts.max_uct import MaxUCT


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def build_data(config, dataset_name):
    data_builder = get_data_builder(config, dataset_name)
    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds = data_builder.build_ts_data()
    train_dl = get_dataloader(train_ts_ds, is_train=True, config=config)
    val_dl = get_dataloader(val_ts_ds, is_train=False, config=config)
    test_dl = get_dataloader(test_ts_ds, is_train=False, config=config)
    return train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl


def optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl, model_name, to_fit=True):
    assert model_name in ["DeepAR", "TFT"], model_name
    trainer = create_trainer()

    model = None
    if model_name == "TFT":
        study = optimize_tft_hp(config, train_dl, val_dl)
        model = create_tft_model(train_ts_ds, study)
    elif model_name == "DeepAR":
        model = create_deepar_model(train_ts_ds)

    if to_fit:
        trainer = fit(trainer, model, train_dl, val_dl)
        model = get_model_from_trainer(trainer, model_name)
    else:
        model = get_model_from_checkpoint(os.getenv("CHECKPOINT"), model_name)
    return model


if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    model_name = os.getenv("MODEL_NAME")
    to_fit = os.getenv("FIT") != "False"

    config = get_config(dataset_name)
    train_df, val_df, test_df, train_ts_ds, val_ts_ds, test_ts_ds, train_dl, val_dl, test_dl \
        = build_data(config, dataset_name)
    plot_data(config, dataset_name, pd.concat([train_df, val_df, test_df], axis=0))
    save_to_pickle(val_df, config.get("ValDataFramePicklePath"))
    save_to_pickle(test_df, config.get("TestDataFramePicklePath"))
    fitted_model = optimize_hp_and_fit(config, train_ts_ds, train_dl, val_dl, model_name, to_fit)
    evaluate(fitted_model, test_dl)
    # plot_predictions(fitted_model, test_dl, test_df, config, dataset_name)
    # tft_env = gym.make("gym_ad_tft:tft-v0")
    # thts = MaxUCT(tft_env, config)
    # thts.run(test_df)
