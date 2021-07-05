import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet
from data_utils import add_dt_columns
from DataBuilders.data_builder import DataBuilder


class SyntheticDataBuilder(DataBuilder):
    def __init__(self, config):
        super().__init__(config)

    def get_data(self):
        data = generate_ar_data(seasonality=self.config.get("Seasonality"),
                                timesteps=self.config.get("Timesteps"),
                                n_series=self.config.get("Series"),
                                trend=self.config.get("Trend"),
                                noise=self.config.get("Noise"))
        data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
        return data

    @staticmethod
    def preprocess(data):
        data = add_dt_columns(data, ['month', 'day_of_month'])
        return data

    def define_ts_ds(self, train_df):
        synthetic_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=["month", "day_of_month"],
            # target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
        )
        return synthetic_train_ts_ds