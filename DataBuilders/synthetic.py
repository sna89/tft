import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from data_utils import add_dt_columns
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN
import numpy as np
import os


PROB = [0.05, 0.075, 0.1, 0.1, 0.175, 0.175, 0.1, 0.1, 0.075, 0.05]


def create_synthetic_dataset(config, full_path):
    num_series = config.get("NumSeries")
    num_sub_series = config.get("NumSubSeries")
    timesteps_per_sub_series = config.get("TimestepsPerSubSeries")

    seasonalities = np.random.choice(np.arange(10, 15, 0.5), size=num_sub_series, replace=True)
    trends = np.random.choice(np.arange(-0.5, 0.5, 0.1), size=num_sub_series, replace=True, p=PROB)
    data1 = create_synthetic_data(num_series, num_sub_series, timesteps_per_sub_series, seasonalities, trends)
    data2 = create_synthetic_data(num_series, num_sub_series, timesteps_per_sub_series, seasonalities / 2, trends)
    data = data1.copy()
    data["value"] = data1["value"] + data2["value"]
    data[DATETIME_COLUMN] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "h")
    data.to_csv(full_path)
    return data


def create_synthetic_data(num_series, num_sub_series, timesteps_per_sub_series, seasonalities, trends):
    start_idx = 0
    start_values = [0] * num_series
    dfs = []

    for i in range(num_sub_series):
        data = generate_ar_data(seasonality=seasonalities[i],
                                timesteps=timesteps_per_sub_series,
                                n_series=num_series,
                                trend=trends[i],
                                noise=0.25,
                                level=1)

        sub_dfs = []
        for n_series, key in enumerate(range(num_series)):
            sub_data = data[data['series'] == key].copy()

            sub_data["time_idx"] = sub_data["time_idx"] + start_idx
            sub_data["value"] = sub_data["value"] + start_values[n_series]
            start_values[n_series] = sub_data["value"].mean()

            sub_dfs.append(sub_data)

        start_idx += timesteps_per_sub_series
        data = pd.concat(sub_dfs, axis=0)
        dfs.append(data)

    data = pd.concat(dfs, axis=0)
    return data


class SyntheticDataBuilder(DataBuilder):
    def __init__(self, config):
        super().__init__(config)

    def build_data(self):
        path = self.config.get("Path")
        filename = self.config.get("Filename")
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            data = pd.read_csv(full_path)
            data = data.drop(columns=['Unnamed: 0'], axis=1)
        else:
            data = create_synthetic_dataset(self.config, full_path)
        return data

    def preprocess(self, data):
        data[DATETIME_COLUMN] = pd.to_datetime(data[DATETIME_COLUMN])
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data[self.config.get("GroupKeyword")] = data[self.config.get("GroupKeyword")].astype(str).astype("category")
        return data

    def define_regression_ts_ds(self, train_df):
        ts_ds = TimeSeriesDataSet(
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
            categorical_encoders={**{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                                     self.config.get("DatetimeAdditionalColumns")}},
        )
        return ts_ds