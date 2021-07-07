import gym
from gym import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pytorch_forecasting import TimeSeriesDataSet
from data_utils import get_dataloader
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import random


@dataclass
class State:
    series: int
    temperature: float = 0
    history: List[float] = field(default_factory=list)


@dataclass
class EnvState:
    env_state: List[State] = field(default_factory=list)
    steps_from_alert: int = -1


class AdTftEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, tft_model, val_ts_ds, val_df):
        self.config = config

        self.tft_model = tft_model
        self.model_pred_len = self.config.get("PredictionLength")
        self.model_enc_len = self.config.get("EncoderLength")

        self.val_ts_ds = val_ts_ds
        self.val_df = val_df
        self.last_date = self._get_last_date(self.val_df)
        self.last_time_idx = self._get_last_time_idx(self.val_df)

        self.reward_false_alert = config["Env"]["Rewards"]["FalseAlert"]
        self.reward_missed_alert = config["Env"]["Rewards"]["MissedAlert"]
        self.reward_good_alert = config["Env"]["Rewards"]["GoodAlert"]

        self.alert_prediction_steps = config["Env"]["AlertPredictionSteps"]
        self.max_steps_from_alert = self.alert_prediction_steps + 1
        self.min_steps_from_alert = 1
        self.anomaly_bounds = config["AnomalyConfig"]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(len(self.anomaly_bounds), 1))

        self.current_state = EnvState()
        self.reset()

    def step(self, action):
        prediction_df = self.build_prediction_df()
        raw_predictions, x = self.tft_model.predict(prediction_df, mode="raw", return_x=True)
        predictions = self.sample_from_predictions(raw_predictions)
        prob = self.calc_sample_probability(raw_predictions)
        reward = self.calc_reward(predictions)
        #return next_state, reward, terminal, prob

    @staticmethod
    def sample_from_predictions(raw_predictions):
        qauntile_predictions = raw_predictions["prediction"]
        num_quantiles = len(qauntile_predictions[0][0])
        quantile_idx = random.choice(list(range(num_quantiles)))
        predictions = list(map(lambda x: x[0][0][quantile_idx], qauntile_predictions.unsqueeze(1)))
        return predictions

    def calc_reward(self, predictions):
        return 0

    def calc_sample_probability(self, raw_predictions):
        return 0

    def reset(self):
        self.current_state.env_state.clear()
        last_sample_df = self.val_df[self.val_df.time_idx == self.val_df.time_idx.max()]
        for idx, sample in last_sample_df.iterrows():
            state = State(sample['series'], sample['value'], [])
            self.current_state.env_state.append(state)
        self.current_state.steps_from_alert = self.max_steps_from_alert
        return self.current_state

    def render(self, mode='human'):
        pass

    def build_prediction_df(self):
        prediction_df = self.val_df
        for current_series_info in self.current_state.env_state:
            for idx, value in enumerate(current_series_info.history, start=1):
                series = current_series_info.series
                prediction_df = self._add_history_sample_to_prediction_df(prediction_df, value, series, idx)

        prediction_df = self._add_dummy_sample_to_prediction_df(prediction_df)
        prediction_df.reset_index(drop=True, inplace=True)
        prediction_df.sort_values(by=['series', 'time_idx'], axis=0, inplace=True)
        return prediction_df

    def _add_history_sample_to_prediction_df(self, prediction_df, value, series, idx):
        data = {'series': series,
                'value': value,
                'date': self.last_date + datetime.timedelta(days=idx),
                'time_idx': self.last_time_idx + idx
                }
        data_df = pd.DataFrame.from_dict(data)
        data_df['day_of_month'] = data_df.date.dt.day.astype(str).astype("category")
        data_df['month'] = data_df.date.dt.month.astype(str).astype("category")
        prediction_df = pd.concat([prediction_df, data_df], axis=0)
        return prediction_df

    def _add_dummy_sample_to_prediction_df(self, prediction_df):
        last_time_idx = self._get_last_time_idx(prediction_df)
        dummy_data = prediction_df[lambda x: x.time_idx == last_time_idx]

        dummy_time_idx = last_time_idx + 1
        dummy_data['time_idx'] = dummy_time_idx

        dummy_date = pd.to_datetime(self._get_last_date(prediction_df)) + timedelta(days=1)
        dummy_data['date'] = dummy_date
        dummy_data['day_of_month'] = dummy_data.date.dt.day.astype(str).astype("category")
        dummy_data['month'] = dummy_data.date.dt.month.astype(str).astype("category")

        prediction_df = pd.concat([prediction_df, dummy_data], axis=0)
        return prediction_df

    @staticmethod
    def _get_last_date(df):
        last_date = df[df.time_idx == df.time_idx.max()]['date'].unique()[0]
        return last_date

    @staticmethod
    def _get_last_time_idx(df):
        return df.time_idx.max()

    # def _build_current_state_df(self):
    #     simulated_test_df = pd.DataFrame()
    #     current_date = self.init_test_df[self.init_test_df.time_idx == self.current_time_idx]['date'].unique()
    #     for series, state in enumerate(self.current_state.env_state):
    #         history = state.history
    #         for time_idx_delta, value in enumerate(history, start=1):
    #             data = {}
    #             data['series'] = series
    #             data['time_idx'] = self.current_time_idx + time_idx_delta
    #             data['value'] = value
    #             data['date'] = current_date + timedelta(days=1)
    #             data['day_of_month'] = data['date'].dt.day
    #             data['month'] = data['date'].dt.month
    #             data_df = pd.DataFrame.from_dict(data)
    #             data['day_of_month'] = data.date.dt.day.astype(str).astype("category")
    #             data['month'] = data.date.dt.month.astype(str).astype("category")
    #             simulated_test_df = pd.concat([simulated_test_df, data_df], axis=0)
    #     current_state_df = pd.concat([self.init_test_df, simulated_test_df], axis=0)
    #     return current_state_df

    # def _get_first_test_time_idx(self):
    #     return self.val_ts_ds.decoded_index.time_idx_last.min() - 1
    #
    # def _get_initial_test_df(self, val_df):
    #     return val_df.loc[val_df.time_idx.isin(list(range(self.current_time_idx - self.model_enc_len - self.model_pred_len, self.current_time_idx + 1)))]
