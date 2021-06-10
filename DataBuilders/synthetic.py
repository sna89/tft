import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet
from config import DataConst
from data_utils import add_dt_columns
from DataBuilders.data_builder import DataBuilder


class Params:
    SERIES = 1
    SEASONALITY = 30
    TREND = 2


class SyntheticDataBuilder(DataBuilder):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_data():
        data = generate_ar_data(seasonality=Params.SEASONALITY,
                                timesteps=600,
                                n_series=Params.SERIES,
                                trend=Params.TREND,
                                noise=0.05)
        data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
        return data

    @staticmethod
    def preprocess(data):
        data = add_dt_columns(data, ['month', 'day_of_month'])
        return data

    @staticmethod
    def define_ts_ds(train_df):
        synthetic_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            min_encoder_length=DataConst.ENCODER_LENGTH,
            max_encoder_length=DataConst.ENCODER_LENGTH,
            min_prediction_length=DataConst.PREDICTION_LENGTH,
            max_prediction_length=DataConst.PREDICTION_LENGTH,
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=["month", "day"],
            # target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
        )
        return synthetic_train_ts_ds