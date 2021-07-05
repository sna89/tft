import pandas as pd
from plot import plot_predictions, plot_synthetic_data
from config import get_config
from Models.tft import create_trainer, create_tft_model, fit, get_fitted_model, optimize_tft_hp
from Models.attention_distance import calc_attention_dist
from evaluation import evaluate, evaluate_base_model
import warnings
from DataBuilders.electricity import ElectricityDataBuilder
from data_utils import get_dataloader
import os
from data_factory import get_data_builder

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    config = get_config(dataset_name)
    data_builder = get_data_builder(config, dataset_name)

    train_df, val_df, test_df, train, val, test = data_builder.build_ts_data()
    # plot_synthetic_data(config, pd.concat([train_df, val_df, test_df], axis=0))
    train_dataloader = get_dataloader(train, is_train=True, config=config)
    val_dataloader = get_dataloader(val, is_train=False, config=config)
    test_dataloader = get_dataloader(test, is_train=False, config=config)

    trainer = create_trainer()
    study = optimize_tft_hp(config, train_dataloader, val_dataloader)
    tft_model = create_tft_model(train, study)
    tft_model = create_tft_model(train, study)
    trainer = fit(trainer, tft_model, train_dataloader, val_dataloader)

    model = get_fitted_model(trainer)
    plot_predictions(model, test_dataloader, test_df, config, dataset_name)
    evaluate(model, val_dataloader)
    evaluate(model, test_dataloader)
