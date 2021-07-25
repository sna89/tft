import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet
from data_utils import add_dt_columns
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN


class SyntheticDataBuilder(DataBuilder):
    def __init__(self, config):
        super().__init__(config)

    def get_data(self):
        data = generate_ar_data(seasonality=self.config.get("Seasonality"),
                                timesteps=self.config.get("Timesteps"),
                                n_series=self.config.get("Series"),
                                trend=self.config.get("Trend"),
                                noise=self.config.get("Noise"))
        data[DATETIME_COLUMN] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
        return data

    def preprocess(self, data):
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data[self.config.get("GroupKeyword")] = data[self.config.get("GroupKeyword")].astype(str).astype("category")
        return data

    def define_ts_ds(self, train_df):
        synthetic_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=[self.config.get("ValueKeyword")],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            # target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
        )
        return synthetic_train_ts_ds