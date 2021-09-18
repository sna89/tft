import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime
import datetime
from env_thts_common import get_reward, build_next_state, EnvState, State, get_group_names
import os
from config import DATETIME_COLUMN
from data_utils import add_dt_columns, reverse_key_value_mapping, get_group_idx_mapping
from Models.trainer import get_prediction_mode


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, model, val_df, test_df):
        self.env_name = "simulation"
        self.config = config

        self.model = model
        self.model_pred_len = self.config.get("Data").get("PredictionLength")
        self.model_enc_len = self.config.get("Data").get("EncoderLength")

        self.val_df = val_df
        self.test_df = test_df
        self.group_idx_mapping = get_group_idx_mapping(self.config, self.model, self.test_df)
        self.last_date = self._get_last_date(self.val_df)
        self.last_time_idx = self._get_last_time_idx(self.val_df)

        self.alert_prediction_steps = config.get("Env").get("AlertMaxPredictionSteps")
        self.min_steps_from_alert = config.get("Env").get("AlertMinPredictionSteps")
        self.max_steps_from_alert = self.alert_prediction_steps + 1
        self.restart_steps = config.get("Env").get("RestartSteps")
        self.max_restart_steps = self.restart_steps + 1

        self.num_series = self._get_num_series()
        self.num_quantiles = self._get_num_quantiles()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.num_series, 1))

        self.current_state = EnvState()
        self.reset()

    def step(self, action: int):
        assert action in [0, 1], "Action must be part of action space"
        assert (all(series_state.steps_from_alert < self.max_steps_from_alert
                    for series_state in self.current_state.env_state)
                and action == 0) or any((series_state.steps_from_alert == self.max_steps_from_alert
                                         for series_state in self.current_state.env_state))

        prediction = self._predict_next_state()

        group_names = get_group_names(self.group_idx_mapping)
        action = {group_name: action for group_name in group_names}

        next_state, terminal_states, _ = build_next_state(self.env_name,
                                                          self.config,
                                                          self.current_state,
                                                          group_names,
                                                          prediction,
                                                          self.max_steps_from_alert,
                                                          self.max_restart_steps,
                                                          action)

        reward = get_reward(self.env_name,
                            self.config,
                            group_names,
                            terminal_states,
                            self.current_state,
                            action)

        return next_state, reward

    def _sample_from_prediction(self, model_prediction):
        quantile_idx_list = np.random.randint(low=0, high=self.num_quantiles, size=self.num_series)
        prediction = [quantile_prediction[0][0][quantile_idx_list[idx]] for idx, quantile_prediction in
                      enumerate(model_prediction.unsqueeze(1))]
        return prediction

    def reset(self):
        self.current_state.env_state.clear()
        last_sample_df = self.val_df[self.val_df.time_idx == self.val_df.time_idx.max()]
        for idx, sample in last_sample_df.iterrows():
            state = State(sample[self.config.get("Data").get("GroupKeyword")],
                          self.max_steps_from_alert,
                          self.max_restart_steps,
                          sample[self.config.get("Data").get("ValueKeyword")],
                          [])
            self.current_state.env_state.append(state)
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
        for dt_column in self.config.get("Data").get("DatetimeAdditionalColumns"):
            prediction_df[dt_column] = prediction_df[dt_column].astype(str).astype("category")
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def _add_sample_to_data(self, new_data, value, series, idx_diff):
        data = {self.config.get("Data").get("GroupKeyword"): series,
                self.config.get("Data").get("ValueKeyword"): value,
                DATETIME_COLUMN: self.last_date + datetime.timedelta(days=idx_diff),
                'time_idx': self.last_time_idx + idx_diff
                }
        add_dt_columns(data, self.config.get("Data").get("DatetimeAdditionalColumns"))
        new_data.append(data)
        return new_data

    def _add_dummy_sample_to_data(self, new_data):
        if not new_data:
            dummy_data = self.val_df[lambda x: x.time_idx == self._get_last_time_idx(self.val_df)].to_dict('records')
        else:
            dummy_data = new_data[-self.num_series:]
        idx_diff = len(self.current_state.env_state[0].history) + 1

        for sample in dummy_data:
            series = sample[self.config.get("Data").get("GroupKeyword")]
            value = sample[self.config.get("Data").get("ValueKeyword")]
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
        last_date = df[df.time_idx == df.time_idx.max()][DATETIME_COLUMN].unique()[0]
        return pd.to_datetime(last_date)

    @staticmethod
    def _get_last_time_idx(df):
        return df.time_idx.max()

    def _get_num_quantiles(self):
        return len(self.model.hparams.loss.quantiles)

    def _get_num_series(self):
        return len(list(self.val_df[self.config.get("Data").get("GroupKeyword")].unique()))

    def _predict_next_state(self):
        prediction_df = self._build_prediction_df()
        mode = get_prediction_mode(self.config)
        model_prediction, x = self.model.predict(prediction_df, mode=mode, return_x=True)
        if isinstance(model_prediction, dict) and "prediction" in model_prediction:
            model_prediction = model_prediction["prediction"]
        prediction = self._sample_from_prediction(model_prediction)
        idx_group_mapping = reverse_key_value_mapping(self.group_idx_mapping)
        prediction = {idx_group_mapping[idx.item()]: value for idx, value in zip(x["groups"], prediction)}
        return prediction


