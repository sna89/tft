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

    def __init__(self, config, tft_model, val_df, test_df):
        self.config = config

        self.tft_model = tft_model
        self.model_pred_len = self.config.get("PredictionLength")
        self.model_enc_len = self.config.get("EncoderLength")

        self.val_df = val_df
        self.test_df = test_df
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
        assert action in self.action_space, "Action must be part of action space"
        assert (
                       self.min_steps_from_alert <= self.current_state.steps_from_alert < self.max_steps_from_alert and action == 0) \
               or (self.current_state.steps_from_alert == self.max_steps_from_alert and action in self.action_space), \
            "{}_{}".format(self.current_state.steps_from_alert, action)

        steps_from_alert = self.current_state.steps_from_alert
        prediction = self._predict_next_state()
        if action == 1 or self.current_state.steps_from_alert < self.max_steps_from_alert:
            steps_from_alert -= 1
        reward, terminal = self._get_reward_and_terminal(prediction, steps_from_alert)
        next_state = self._build_next_state(prediction, steps_from_alert)
        prob = 1 / float(self._get_num_quantiles())
        return next_state, reward, terminal, prob

    def _sample_from_prediction(self, raw_prediction):
        quantile_prediction = raw_prediction["prediction"]
        num_quantiles = self._get_num_quantiles()
        quantile_idx = random.choice(list(range(num_quantiles)))
        prediction = list(map(lambda x: x[0][0][quantile_idx], quantile_prediction.unsqueeze(1)))
        return prediction

    def _get_reward_and_terminal(self, prediction, steps_from_alert):
        reward = 0
        terminal = False
        for num_series in range(len(prediction)):
            bounds = self.config.get("AnomalyConfig").get("series_{}".format(num_series))
            lb, hb = bounds.values()
            series_prediction = prediction[num_series]
            if self._is_missed_alert(lb, hb, series_prediction, steps_from_alert):
                reward += self.reward_missed_alert
                terminal = True
            if self._is_false_alert(lb, hb, series_prediction, steps_from_alert):
                reward += self.reward_false_alert
            if self._is_good_alert(lb, hb, series_prediction, steps_from_alert):
                reward += self._calc_good_alert_reward(steps_from_alert)
                terminal = True
        return reward, terminal

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

    def _build_prediction_df(self):
        prediction_df = self.val_df
        for current_series_info in self.current_state.env_state:
            for idx, value in enumerate(current_series_info.history[1:], start=1):
                series = current_series_info.series
                prediction_df = self._add_sample_to_prediction_df(prediction_df, value, series, idx)

        prediction_df = self._add_current_state_to_prediction_df(prediction_df)
        prediction_df = self._add_dummy_sample_to_prediction_df(prediction_df)

        prediction_df.reset_index(drop=True, inplace=True)
        prediction_df.sort_values(by=['series', 'time_idx'], axis=0, inplace=True)
        return prediction_df

    def _add_sample_to_prediction_df(self, prediction_df, value, series, idx_diff):
        data = {'series': series,
                'value': value,
                'date': self.last_date + datetime.timedelta(days=idx_diff),
                'time_idx': self.last_time_idx + idx_diff
                }
        data_df = pd.DataFrame(data, index=[0])
        data_df['day_of_month'] = data_df.date.dt.day.astype(str).astype("category")
        data_df['month'] = data_df.date.dt.month.astype(str).astype("category")
        prediction_df = pd.concat([prediction_df, data_df], axis=0)
        return prediction_df

    def _add_dummy_sample_to_prediction_df(self, prediction_df):
        last_time_idx = self._get_last_time_idx(prediction_df)
        dummy_data = prediction_df[lambda x: x.time_idx == last_time_idx]
        idx_diff = len(self.current_state.env_state[0].history) + 1

        num_series = list(prediction_df['series'].unique())
        for series in num_series:
            value = float(dummy_data[dummy_data.series == series]['value'])
            prediction_df = self._add_sample_to_prediction_df(prediction_df, value, series, idx_diff)

        return prediction_df

    def _add_current_state_to_prediction_df(self, prediction_df):
        idx_diff = len(self.current_state.env_state[0].history)
        if idx_diff > 0:
            for series_state in self.current_state.env_state:
                prediction_df = self._add_sample_to_prediction_df(prediction_df,
                                                                  series_state.temperature,
                                                                  series_state.series,
                                                                  idx_diff)
        return prediction_df

    @staticmethod
    def _get_last_date(df):
        last_date = df[df.time_idx == df.time_idx.max()]['date'].unique()[0]
        return pd.to_datetime(last_date)

    @staticmethod
    def _get_last_time_idx(df):
        return df.time_idx.max()

    def _get_num_quantiles(self):
        return self.tft_model.output_layer.out_features

    def _is_missed_alert(self, lb, hb, prediction, steps_from_alert):
        if (prediction < lb or prediction > hb) and (steps_from_alert == self.max_steps_from_alert):
            return True
        return False

    def _is_false_alert(self, lb, hb, prediction, steps_from_alert):
        if (lb <= prediction <= hb) and (steps_from_alert == 1):
            return True
        return False

    def _is_good_alert(self, lb, hb, prediction, steps_from_alert):
        if (prediction < lb or prediction > hb) and (steps_from_alert < self.max_steps_from_alert):
            return True
        return False

    def _calc_good_alert_reward(self, steps_from_alert):
        return self.reward_good_alert * (self.max_steps_from_alert - steps_from_alert)

    def _predict_next_state(self):
        prediction_df = self._build_prediction_df()
        raw_prediction, x = self.tft_model.predict(prediction_df, mode="raw", return_x=True)
        prediction = self._sample_from_prediction(raw_prediction)
        return prediction

    def _build_next_state(self, prediction, steps_from_alert):
        if steps_from_alert == 0:
            steps_from_alert = self.max_steps_from_alert

        next_state = EnvState()
        next_state.steps_from_alert = steps_from_alert
        for series in range(len(prediction)):
            series_current_state = self.current_state.env_state[series]
            next_state_series_history = series_current_state.history + [series_current_state.temperature]
            next_state_series_temperature = prediction[series].item()
            next_state_series = State(series,
                                      next_state_series_temperature,
                                      next_state_series_history)
            next_state.env_state.append(next_state_series)
        return next_state
