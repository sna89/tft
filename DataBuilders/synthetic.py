import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from data_utils import add_dt_columns
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN
import numpy as np
import os


def create_correlated_normal_series(n_series, num_correlated):
    s_1 = np.random.normal(loc=1, scale=0.5, size=1) * np.ones(num_correlated)
    s_2 = np.random.normal(size=n_series - num_correlated)
    s = np.concatenate((s_1, s_2))
    return s


def create_correlated_uniform_series(n_series, num_correlated):
    s_1 = np.random.uniform(low=0, high=1, size=1) * np.ones(num_correlated)
    s_2 = np.random.uniform(low=-1, high=1, size=n_series - num_correlated)
    s = np.concatenate((s_1, s_2))
    return s


def create_levels(level, n_series, num_correlated):
    if not num_correlated:
        levels = level * np.random.uniform(low=-1, high=1, size=n_series)[:, None]
    else:
        levels = level * create_correlated_uniform_series(n_series, num_correlated)[:, None]
    return levels


def create_trend(n_series, num_sub_series, num_correlated, timesteps, type_="linear"):
    assert type_ in ["linear", "quadratic"]

    trends_list = []
    for i in range(num_sub_series):
        trends = None
        if not num_correlated:
            if type_ == "linear":
                trends = np.random.normal(size=n_series)[:, None] / timesteps
            elif type_ == "quadratic":
                trends = np.random.normal(size=n_series)[:, None] / timesteps ** 2
        else:
            if type_ == "linear":
                trends = create_correlated_normal_series(n_series, num_correlated)[:, None] / timesteps
            elif type_ == "quadratic":
                trends = create_correlated_normal_series(n_series, num_correlated)[:, None] / timesteps ** 2
        trends_list.append(trends)
    return trends_list


def create_seasonality_f(n_series, timesteps):
    seasonality_f = np.random.randint(low=timesteps // 20, high=timesteps // 2, size=n_series)[:, None]
    return seasonality_f


def create_seasonality_amplitude(n_series):
    seasonality_amp = np.random.normal(loc=1, scale=0.5, size=n_series)[:, None]
    return seasonality_amp


def create_range_series(timesteps):
    x = np.arange(timesteps)[None, :]
    return x


def add_linear_trend_to_series(x, trend, linear_trends):
    return x * linear_trends * trend


def add_seasonality_to_series(x, timesteps, seasonality_amp, seasonality_f):
    return seasonality_amp * np.sin(2 * np.pi * seasonality_f * x / timesteps)


def add_level_and_noise_to_series(x, levels, noise):
    series = levels * x + noise * np.random.normal(size=x.shape)
    return series


def generate_parameters(
        n_series: int = 10,
        num_sub_series: int = 10,
        timesteps_sub_series: int = 400,
        level: float = 1.0,
        num_correlated: int = 0):
    num_correlated = max(min(n_series, num_correlated), 0)
    linear_trends_list = create_trend(n_series, num_sub_series, num_correlated, timesteps_sub_series, type_="linear")
    levels = create_levels(level, n_series, num_correlated)
    seasonality_amp = create_seasonality_amplitude(n_series)
    seasonality_f = create_seasonality_f(n_series, timesteps_sub_series)

    return linear_trends_list, levels, seasonality_amp, seasonality_f


def create_data_partial(config,
                        timesteps_sub_series,
                        trend,
                        trends,
                        seasonality_amp,
                        seasonality_f,
                        levels,
                        noise):
    series = create_range_series(timesteps_sub_series)
    series = add_linear_trend_to_series(series, trend, trends)
    series = add_seasonality_to_series(series, timesteps_sub_series, seasonality_amp, seasonality_f)
    series = add_level_and_noise_to_series(series, levels, noise)

    data = (
        pd.DataFrame(series)
            .stack()
            .reset_index()
            .rename(columns={"level_0": config.get("GroupKeyword"),
                             "level_1": "time_idx",
                             0: config.get("ValueKeyword")})
    )

    return data


def create_synthetic_dataset(config,
                             num_series,
                             num_sub_series,
                             timesteps_sub_series,
                             trends_list,
                             seasonality_amp,
                             seasonality_f,
                             levels,
                             noise,
                             trend):
    start_idx = 0
    start_values = [0] * num_series
    dfs = []

    for i in range(num_sub_series):
        data = create_data_partial(config,
                                   timesteps_sub_series,
                                   trend,
                                   trends_list[i],
                                   seasonality_amp,
                                   seasonality_f,
                                   levels,
                                   noise)

        sub_dfs = []
        keys = list(pd.unique(data[config.get("GroupKeyword")]))
        for i, key in enumerate(keys):
            sub_data = data[data[config.get("GroupKeyword")] == key].copy()

            sub_data["time_idx"] = sub_data["time_idx"] + start_idx
            sub_data[config.get("ValueKeyword")] = sub_data[config.get("ValueKeyword")] + start_values[i]
            start_values[i] = sub_data[sub_data["time_idx"] == sub_data["time_idx"].max()][config.get("ValueKeyword")].values

            sub_dfs.append(sub_data)

        start_idx += timesteps_sub_series
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
            trends_list, levels, seasonality_amp, seasonality_f = generate_parameters(self.config.get("NumSeries"),
                                                                                      self.config.get("NumSubSeries"),
                                                                                      self.config.get(
                                                                                          "TimeStepsSubSeries"),
                                                                                      num_correlated=self.config.get(
                                                                                          "NumCorrelatedSeries"),
                                                                                      level=self.config.get("Level"))
            data1 = create_synthetic_dataset(self.config,
                                             self.config.get("NumSeries"),
                                             self.config.get("NumSubSeries"),
                                             self.config.get("TimeStepsSubSeries"),
                                             trends_list,
                                             seasonality_amp,
                                             seasonality_f,
                                             levels,
                                             noise=self.config.get("Noise"),
                                             trend=self.config.get("Trend"))

            data2 = create_synthetic_dataset(self.config,
                                             self.config.get("NumSeries"),
                                             self.config.get("NumSubSeries"),
                                             self.config.get("TimeStepsSubSeries"),
                                             trends_list,
                                             seasonality_amp,
                                             seasonality_f // 2,
                                             levels,
                                             noise=self.config.get("Noise"),
                                             trend=self.config.get("Trend"))

            data3 = create_synthetic_dataset(self.config,
                                             self.config.get("NumSeries"),
                                             self.config.get("NumSubSeries"),
                                             self.config.get("TimeStepsSubSeries"),
                                             trends_list,
                                             seasonality_amp,
                                             seasonality_f // 14,
                                             levels,
                                             noise=self.config.get("Noise"),
                                             trend=self.config.get("Trend"))
            data = data1.copy()
            target_column = self.config.get("ValueKeyword")
            data[target_column] = data1[target_column] + data2[target_column] + data3[target_column]
            data.reset_index(drop=True, inplace=True)
            data.to_csv(full_path)
        return data

    def preprocess(self, data):
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
            time_varying_known_categoricals=[],
            # target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None
        )
        return ts_ds
