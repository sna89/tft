import pandas as pd
from plot import plot_predictions
from config import get_config
from Models.tft import create_trainer, create_tft_model, fit, get_fitted_model
from Models.attention_distance import calc_attention_dist
from evaluation import evaluate, evaluate_base_model
import warnings
from DataBuilders.electricity import ElectricityDataBuilder
from data_utils import get_dataloader
import os

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    config = get_config(dataset_name)
    elec_data_builder = ElectricityDataBuilder(config['Train'].get('TrainRatio'),
                                               config['Train'].get('ValRatio'),
                                               config.get('EncoderLength'),
                                               config.get('PredictionLength'),
                                               config.get('Path'),
                                               config.get("NumGroups"),
                                               config.get("ProcessedDfColumnNames"),
                                               config.get("StartDate"),
                                               config.get("EndDate"))

    train_df, val_df, test_df, train, val, test = elec_data_builder.build_ts_data()
    train_dataloader = get_dataloader(train, is_train=True, config=config)
    val_dataloader = get_dataloader(val, is_train=False, config=config)
    test_dataloader = get_dataloader(test, is_train=False, config=config)

    # print(evaluate_base_model(val_dataloader))
    # print(evaluate_base_model(test_dataloader))

    trainer = create_trainer()
    tft_model = create_tft_model(train)
    # trainer = fit(trainer, tft_model, train_dataloader, val_dataloader)

    model = get_fitted_model(trainer)
    plot_predictions(model, test_dataloader, test_df, config, dataset_name)
    evaluate(model, val_dataloader)
    evaluate(model, test_dataloader)

    # calc_attention_dist(model, test_dataloader, test_df)

    # print(evaluate_base_model(val_dataloader))
    # data = data[(data.agency == 'Agency_25') & (data.sku == 'SKU_03')]
    # print(data)
    # plot_volume_by_group(data, 'Agency_25')
