import pandas as pd
import numpy as np


def create_correlated_from_random_choice(n_series, num_correlated, option_list):
    s_1 = np.random.choice(a=option_list, size=1) * np.ones(num_correlated)
    s_2 = np.random.choice(a=option_list, size=n_series - num_correlated)
    s = np.concatenate((s_1, s_2))[:, None]
    return s


def create_correlated_normal_series(n_series, num_correlated, loc=0, scale=1):
    s_1 = np.random.normal(loc=loc, scale=scale, size=1) * np.ones(num_correlated)
    s_2 = np.random.normal(loc=loc, scale=scale, size=n_series - num_correlated)
    s = np.concatenate((s_1, s_2))
    return s


def create_correlated_uniform_series(n_series, num_correlated, low=-0.25, high=0.25):
    s_1 = np.random.uniform(low=0, high=high, size=1) * np.ones(num_correlated)
    s_2 = np.random.uniform(low=low, high=high, size=n_series - num_correlated)
    s = np.concatenate((s_1, s_2))
    return s


def create_levels(level, n_series, num_correlated):
    low = -1.5
    high = 1.5
    if not num_correlated:
        levels = level * np.random.uniform(low=low, high=high, size=n_series)[:, None]
    else:
        levels = level * create_correlated_uniform_series(n_series, num_correlated, low, high)[:, None]
    return levels


def create_trend(n_series, num_sub_series, num_correlated, timesteps, type_="linear"):
    assert type_ in ["linear", "quadratic", "sin"]
    loc = 0
    scale = 1

    trends_list = []
    for i in range(num_sub_series):
        trends = None
        if not num_correlated:
            if type_ == "linear":
                trends = np.random.normal(loc=loc, scale=scale, size=n_series)[:, None] / timesteps
            elif type_ == "quadratic":
                trends = np.random.normal(size=n_series)[:, None] / timesteps ** 2
        else:
            if type_ == "linear":
                trends = create_correlated_normal_series(n_series, num_correlated, loc=loc, scale=scale)[:,
                         None] / timesteps
            elif type_ == "quadratic":
                trends = create_correlated_normal_series(n_series, num_correlated)[:, None] / timesteps ** 2
        trends_list.append(trends)
    return trends_list


def create_seasonality_f(n_series, timesteps, cycle_length_low, cycle_length_high, num_correlated):
    seasonality_low = timesteps // cycle_length_high
    seasonality_high = timesteps // cycle_length_low
    option_list = list(range(seasonality_low, seasonality_high + 1, 2))
    seasonality_f = create_correlated_from_random_choice(n_series, num_correlated, option_list)
    return seasonality_f


def create_seasonality_amplitude(n_series):
    seasonality_amp = np.random.normal(loc=2, scale=0.3, size=n_series)[:, None]
    return seasonality_amp


def create_range_series(timesteps):
    x = np.arange(timesteps)[None, :]
    return x


def add_linear_trend_to_series(x, trend, linear_trends):
    return x * linear_trends * trend


def add_quadratic_trend_to_series(x, trend, quadratic_trends):
    return x ** 2 * quadratic_trends * trend


def add_trend_to_series(x, trend, linear_trends, quadratic_trends):
    x = add_linear_trend_to_series(x, trend, linear_trends)
    x = add_quadratic_trend_to_series(x, trend, quadratic_trends)
    return x


def add_seasonality_to_series(trend, range_series, timesteps, seasonality_amp, seasonality_f):
    return trend + seasonality_amp * np.sin(2 * np.pi * seasonality_f * range_series / timesteps)


def add_level_and_noise_to_series(x, levels, noise):
    series = levels * x + noise * np.random.normal(loc=0, scale=0.3, size=x.shape)
    return series


def create_data_partial(timesteps_sub_series,
                        trend,
                        linear_trends_list,
                        seasonality_amp,
                        seasonality_f,
                        levels,
                        noise):
    range_series = create_range_series(timesteps_sub_series)
    trend = add_linear_trend_to_series(range_series, trend, linear_trends_list)
    series = add_seasonality_to_series(trend, range_series, timesteps_sub_series, seasonality_amp, seasonality_f)
    series = add_level_and_noise_to_series(series, levels, noise)

    data = (
        pd.DataFrame(series)
            .stack()
            .reset_index()
            .rename(columns={"level_0": "Series", "level_1": "time_idx", 0: "Value"})
    )

    return data


def create_data_all(num_series,
                    num_sub_series,
                    timesteps_sub_series,
                    linear_trends_list,
                    seasonality_amp,
                    seasonality_f,
                    levels,
                    noise,
                    trend):
    start_idx = 0
    start_values = [0] * num_series
    dfs = []

    for i in range(num_sub_series):
        data = create_data_partial(timesteps_sub_series,
                                   trend,
                                   linear_trends_list[i],
                                   seasonality_amp,
                                   seasonality_f,
                                   levels,
                                   noise)

        sub_dfs = []
        keys = list(pd.unique(data['Series']))
        for i, key in enumerate(keys):
            sub_data = data[data['Series'] == key].copy()

            sub_data["time_idx"] = sub_data["time_idx"] + start_idx
            sub_data["Value"] = sub_data["Value"] + start_values[i]
            start_values[i] = sub_data["Value"].mean()
            # start_values[i] = sub_data[sub_data["time_idx"].max() == sub_data["time_idx"]]["Value"].values

            sub_dfs.append(sub_data)

        start_idx += timesteps_sub_series
        data = pd.concat(sub_dfs, axis=0)
        dfs.append(data)

    data = pd.concat(dfs, axis=0)

    return data


def generate_parameters(
        n_series: int = 10,
        num_sub_series: int = 10,
        timesteps_sub_series: int = 400,
        level: float = 1.0,
        num_correlated: int = 0,
        cycle_length_low: int = 10,
        cycle_length_high: int = 100,
):
    num_correlated = max(min(n_series, num_correlated), 0)
    linear_trends_list = create_trend(n_series, num_sub_series, num_correlated, timesteps_sub_series, type_="linear")
    levels = create_levels(level, n_series, num_correlated)
    seasonality_amp = create_seasonality_amplitude(n_series)
    seasonality_f = create_seasonality_f(n_series, timesteps_sub_series, cycle_length_low, cycle_length_high,
                                         num_correlated)

    return linear_trends_list, levels, seasonality_amp, seasonality_f
