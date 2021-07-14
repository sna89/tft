import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime
import datetime
import random
from env_thts_common import get_reward_and_terminal, build_next_state, EnvState, State


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

        self.alert_prediction_steps = config["Env"]["AlertMaxPredictionSteps"]
        self.min_steps_from_alert = config["Env"]["AlertMinPredictionSteps"]
        self.max_steps_from_alert = self.alert_prediction_steps + 1

        self.num_series = self._get_num_series()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.num_series, 1))

        self.current_state = EnvState()
        self.reset()

    def step(self, action):
        assert action in self.action_space, "Action must be part of action space"
        assert (
                       self.min_steps_from_alert < self.current_state.steps_from_alert < self.max_steps_from_alert and action == 0) \
               or (self.current_state.steps_from_alert == self.max_steps_from_alert and action in self.action_space), \
            "{}_{}".format(self.current_state.steps_from_alert, action)

        prediction = self._predict_next_state()
        next_state = build_next_state(self.current_state, prediction, self.max_steps_from_alert, action)
        reward, terminal = get_reward_and_terminal(self.config, prediction, self.current_state.steps_from_alert, action)
        prob = 1 / float(self._get_num_quantiles())
        return next_state, reward, terminal, prob

    def _sample_from_prediction(self, raw_prediction):
        quantile_prediction = raw_prediction["prediction"]
        num_quantiles = self._get_num_quantiles()
        quantile_idx = random.choice(list(range(num_quantiles)))
        prediction = list(map(lambda x: x[0][0][quantile_idx], quantile_prediction.unsqueeze(1)))
        return prediction

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
        new_data = []

        for current_series_info in self.current_state.env_state:
            for idx, value in enumerate(current_series_info.history[1:], start=1):
                series = current_series_info.series
                new_data = self._add_sample_to_data(new_data, value, series, idx)

        new_data = self._add_current_state_to_data(new_data)
        new_data = self._add_dummy_sample_to_data(new_data)

        prediction_df = pd.concat([prediction_df, pd.DataFrame.from_dict(new_data)], axis=0)
        prediction_df['day_of_month'] = prediction_df.date.dt.day.astype(str).astype("category")
        prediction_df['month'] = prediction_df.date.dt.month.astype(str).astype("category")
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def _add_sample_to_data(self, new_data, value, series, idx_diff):
        data = {'series': series,
                'value': value,
                'date': self.last_date + datetime.timedelta(days=idx_diff),
                'time_idx': self.last_time_idx + idx_diff
                }
        data['day_of_month'] = data['date'].day
        data['month'] = data['date'].day
        new_data.append(data)
        return new_data

    def _add_dummy_sample_to_data(self, new_data):
        if not new_data:
            dummy_data = self.val_df[lambda x: x.time_idx == self._get_last_time_idx(self.val_df)].to_dict('records')
        else:
            dummy_data = new_data[-self.num_series:]
        idx_diff = len(self.current_state.env_state[0].history) + 1

        for sample in dummy_data:
            series = sample['series']
            value = sample['value']
            new_data = self._add_sample_to_data(new_data, value, series, idx_diff)

        return new_data

    def _add_current_state_to_data(self, new_data):
        idx_diff = len(self.current_state.env_state[0].history)
        if idx_diff > 0:
            for series_state in self.current_state.env_state:
                new_data = self._add_sample_to_data(new_data,
                                                    series_state.temperature,
                                                    series_state.series,
                                                    idx_diff)
        return new_data

    @staticmethod
    def _get_last_date(df):
        last_date = df[df.time_idx == df.time_idx.max()]['date'].unique()[0]
        return pd.to_datetime(last_date)

    @staticmethod
    def _get_last_time_idx(df):
        return df.time_idx.max()

    def _get_num_quantiles(self):
        return self.tft_model.output_layer.out_features

    def _get_num_series(self):
        return len(list(self.val_df['series'].unique()))

    def _predict_next_state(self):
        prediction_df = self._build_prediction_df()
        raw_prediction, x = self.tft_model.predict(prediction_df, mode="raw", return_x=True)
        prediction = self._sample_from_prediction(raw_prediction)
        return prediction


