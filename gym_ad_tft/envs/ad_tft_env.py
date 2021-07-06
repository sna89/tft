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


@dataclass
class State:
    temperature: float = 0
    steps_from_alert: int = -1
    history: List[float] = field(default_factory=list)


@dataclass
class EnvState:
    env_state: List[State] = field(default_factory=list)


class AdTftEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, tft_model, test_ts_ds, test_df):
        self.config = config

        self.tft_model = tft_model
        self.model_pred_len = self.config.get("PredictionLength")
        self.model_enc_len = self.config.get("EncoderLength")

        self.test_ts_ds = test_ts_ds
        self.current_time_idx = self._get_first_test_time_idx()
        self.init_test_df = self._get_initial_test_df(test_df)

        self.reward_false_alert = config["Rewards"]["FalseAlert"]
        self.reward_missed_alert = config["Rewards"]["MissedAlert"]
        self.reward_good_alert = config["Rewards"]["GoodAlert"]

        self.alert_prediction_steps = config["Env"]["AlertPredictionSteps"]
        self.max_alert_steps = self.alert_prediction_steps + 1
        self.anomaly_bounds = config["AnomalyConfig"]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(len(self.anomaly_bounds), 1))

        self.current_state = EnvState()
        self.reset()

    def step(self, action):
        current_state_df = self._build_current_state_df()
        current_state_ts_ds = TimeSeriesDataSet.from_parameters(self.test_ts_ds.get_parameters(), current_state_df)
        current_state_dl = get_dataloader(current_state_ts_ds, is_train=False, config=self.config)
        raw_prediction, x = self.tft_model.predict(current_state_dl, mode="raw", return_x=True)
        self.tft_model.plot_prediction(x, raw_prediction, idx=0)

    def reset(self):
        self.current_state.env_state.clear()
        test_first_time_idx = self._get_first_test_time_idx()
        sensor_temperature_list = list(self.init_test_df[self.init_test_df.time_idx == test_first_time_idx]['value'])
        for temperature in sensor_temperature_list:
            state = State(temperature, self.max_alert_steps, [])
            self.current_state.env_state.append(state)

    def render(self, mode='human'):
        pass

    def _build_current_state_df(self):
        simulated_test_df = pd.DataFrame()
        current_date = self.init_test_df[self.init_test_df.time_idx == self.current_time_idx]['date'].unique()
        for series, state in enumerate(self.current_state.env_state):
            history = state.history
            for time_idx_delta, value in enumerate(history, start=1):
                data = {}
                data['series'] = series
                data['time_idx'] = self.current_time_idx + time_idx_delta
                data['value'] = value
                data['date'] = current_date + timedelta(days=1)
                data['day_of_month'] = data['date'].dt.day
                data['month'] = data['date'].dt.month
                data_df = pd.DataFrame.from_dict(data)
                data['day_of_month'] = data.date.dt.day.astype(str).astype("category")
                data['month'] = data.date.dt.month.astype(str).astype("category")
                simulated_test_df = pd.concat([simulated_test_df, data_df], axis=0)
        current_state_df = pd.concat([self.init_test_df, simulated_test_df], axis=0)
        return current_state_df

    def _get_first_test_time_idx(self):
        return self.test_ts_ds.decoded_index.time_idx_last.min() - 1

    def _get_initial_test_df(self, test_df):
        return test_df.loc[test_df.time_idx.isin(list(range(self.current_time_idx - self.model_enc_len - self.model_pred_len, self.current_time_idx + 1)))]
