import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime
import datetime
from env_thts_common import get_reward, build_next_state, EnvState, State, get_last_val_time_idx, get_last_val_date, \
    get_steps_from_alert, get_max_steps_from_alert, get_min_steps_from_alert, get_restart_steps, get_max_restart_steps, \
    get_group_names

from config import DATETIME_COLUMN
from data_utils import add_dt_columns, reverse_key_value_mapping, get_group_id_group_name_mapping
from Models.trainer import get_prediction_mode


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, forecasting_model, test_df, test_ts_ds):
        self.env_name = "simulation"
        self.config = config
        self.forecasting_model = forecasting_model

        self.test_df = test_df
        self.test_ts_ds = test_ts_ds
        self.group_id_group_name_mapping = get_group_id_group_name_mapping(self.config, self.test_ts_ds)

        self.last_val_time_idx = get_last_val_time_idx(self.config, self.test_df)
        self.last_val_date = get_last_val_date(self.test_df, self.last_val_time_idx)
        self.init_prediction_df = self.test_df[self.test_df.time_idx <= self.last_val_time_idx]

        self.alert_prediction_steps = get_steps_from_alert(self.config)
        self.min_steps_from_alert = get_min_steps_from_alert(self.config)
        self.max_steps_from_alert = get_max_steps_from_alert(self.config)
        self.restart_steps = get_restart_steps(self.config)
        self.max_restart_steps = get_max_restart_steps(self.config)

        self.num_series = self._get_num_series()
        self.num_quantiles = self._get_num_quantiles()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.num_series, 1))

        self._current_state = EnvState()
        self.reset()

    def step(self, action: int):
        assert action in [0, 1], "Action must be part of action space"
        assert (all(series_state.steps_from_alert < self.max_steps_from_alert
                    for series_state in self.current_state.env_state)
                and action == 0) or any((series_state.steps_from_alert == self.max_steps_from_alert
                                         for series_state in self.current_state.env_state))

        prediction = self._predict_next_state()

        group_names = get_group_names(self.group_id_group_name_mapping)
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
        prediction = [quantile_prediction[0][quantile_idx_list[idx]] for idx, quantile_prediction in
                      enumerate(model_prediction)]
        return prediction

    def reset(self):
        self.current_state.env_state.clear()
        last_sample_df = self.test_df[self.test_df.time_idx == self.last_val_time_idx]
        for idx, sample in last_sample_df.iterrows():
            state = State(sample[self.config.get("GroupKeyword")],
                          self.max_steps_from_alert,
                          self.max_restart_steps,
                          sample[self.config.get("ValueKeyword")],
                          [])
            self.current_state.env_state.append(state)
        return self.current_state

    def render(self, mode='human'):
        pass

    def _build_prediction_df(self):
        new_data = []

        for current_series_info in self.current_state.env_state:
            for idx, value in enumerate(current_series_info.history[1:], start=1):
                series = current_series_info.group
                new_data = self._add_sample_to_data(new_data, value, series, idx)

        new_data = self._add_current_state_to_data(new_data)
        new_data = self._add_dummy_sample_to_data(new_data)

        prediction_df = pd.concat([self.init_prediction_df, pd.DataFrame.from_dict(new_data)], axis=0)
        for dt_column in self.config.get("DatetimeAdditionalColumns", []):
            prediction_df[dt_column] = prediction_df[dt_column].astype(str).astype("category")
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def _add_sample_to_data(self, new_data, value, series, idx_diff):
        data = {self.config.get("GroupKeyword"): series,
                self.config.get("ValueKeyword"): value,
                DATETIME_COLUMN: self.last_val_date + datetime.timedelta(hours=idx_diff) if self.last_val_date else None,
                'time_idx': self.last_val_time_idx + idx_diff
                }

        dt_columns = self.config.get("DatetimeAdditionalColumns", [])
        add_dt_columns(data, dt_columns)
        new_data.append(data)
        return new_data

    def _add_dummy_sample_to_data(self, new_data):
        if not new_data:
            dummy_data = self.test_df[lambda x: x.time_idx == self.last_val_time_idx].to_dict('records')
        else:
            dummy_data = new_data[-self.num_series:]
        idx_diff = len(self.current_state.env_state[0].history) + 1

        for sample in dummy_data:
            series = sample[self.config.get("GroupKeyword")]
            value = sample[self.config.get("ValueKeyword")]
            new_data = self._add_sample_to_data(new_data, value, series, idx_diff)

        return new_data

    def _add_current_state_to_data(self, new_data):
        idx_diff = len(self.current_state.env_state[0].history)
        if idx_diff > 0:
            for series_state in self.current_state.env_state:
                new_data = self._add_sample_to_data(new_data,
                                                    series_state.value,
                                                    series_state.group,
                                                    idx_diff)
        return new_data

    def _get_num_quantiles(self):
        return len(self.forecasting_model.hparams.loss.quantiles)

    def _get_num_series(self):
        return len(list(self.test_df[self.config.get("GroupKeyword")].unique()))

    def _predict_next_state(self):
        prediction_df = self._build_prediction_df()
        prediction_mode = get_prediction_mode()
        model_prediction, x = self.forecasting_model.predict(prediction_df, mode=prediction_mode, return_x=True)
        if (isinstance(model_prediction, dict) and "prediction" in model_prediction) or \
                (isinstance(model_prediction, tuple) and "prediction" in model_prediction.keys()):
            model_prediction = model_prediction["prediction"]
        prediction = self._sample_from_prediction(model_prediction)
        prediction_dict = {self.group_id_group_name_mapping[idx]: value for idx, value in enumerate(prediction)}
        return prediction_dict

    @property
    def current_state(self):
        return self._current_state
