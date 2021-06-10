import pandas as pd
from plot import plot_data, plot_predictions
from constants import HyperParameters, DataConst
from Models.tft import create_trainer, create_tft_model, fit, get_fitted_model
from Models.attention_distance import calc_attention_dist
from evaluation import evaluate, evaluate_base_model
import warnings
import multiprocessing
from DataBuilders.electricity import ElectricityDataBuilder
from DataBuilders.electricity import Params

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
cpu_count = multiprocessing.cpu_count()

if __name__ == '__main__':
    elec_data_builder = ElectricityDataBuilder(Params.TRAIN_RATIO,
                                               Params.VAL_RATIO,
                                               Params.ENCODER_LENGTH,
                                               Params.PREDICTION_LENGTH)
    train_df, val_df, test_df, train, val, test = elec_data_builder.build_ts_data()
    train_dataloader = train.to_dataloader(train=True, batch_size=HyperParameters.BATCH_SIZE, num_workers=cpu_count)
    val_dataloader = val.to_dataloader(train=False, batch_size=HyperParameters.BATCH_SIZE, num_workers=cpu_count)
    test_dataloader = test.to_dataloader(train=False, batch_size=HyperParameters.BATCH_SIZE, num_workers=cpu_count)

    # print(evaluate_base_model(val_dataloader))
    # print(evaluate_base_model(test_dataloader))

    trainer = create_trainer()
    tft_model = create_tft_model(train)
    trainer = fit(trainer, tft_model, train_dataloader, val_dataloader)

    model = get_fitted_model(trainer)
    plot_predictions(model, test_dataloader)
    # print(evaluate(model, val_dataloader))
    # print(evaluate(model, test_dataloader))

    # calc_attention_dist(model, test_dataloader, test_df)

    # print(evaluate_base_model(val_dataloader))
    # data = data[(data.agency == 'Agency_25') & (data.sku == 'SKU_03')]
    # print(data)
    # plot_volume_by_group(data, 'Agency_25')
