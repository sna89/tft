import datetime
from EnvCommon.env_thts_common import get_last_val_time_idx, get_last_val_date, get_num_series
from config import DATETIME_COLUMN
from data_utils import get_group_id_group_name_mapping, add_dt_columns
from config import get_num_quantiles
from utils import get_prediction_mode
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, config, forecasting_mode, test_df, test_ts_ds):
        self.config = config
        self.forecasting_model = forecasting_mode
        self.test_df = test_df
        self.test_ts_ds = test_ts_ds

        self.last_val_time_idx = get_last_val_time_idx(self.config, self.test_df)
        self.last_val_date = get_last_val_date(self.test_df, self.last_val_time_idx)
        self.init_prediction_df = self.test_df[self.test_df.time_idx <= self.last_val_time_idx]

        self.group_id_group_name_mapping = get_group_id_group_name_mapping(self.config, self.test_ts_ds)

        self.num_quantiles = get_num_quantiles()
        self.num_series = get_num_series(self.config, self.test_df)

    def predict(self, current_state):
        prediction_df = self._build_prediction_df(current_state)

        prediction_mode = get_prediction_mode()
        model_prediction, x = self.forecasting_model.predict(prediction_df, mode=prediction_mode, return_x=True)
        if (isinstance(model_prediction, dict) and "prediction" in model_prediction) or \
                (isinstance(model_prediction, tuple) and "prediction" in model_prediction.keys()):
            model_prediction = model_prediction["prediction"]

        prediction_dict = {self.group_id_group_name_mapping[group_id.item()]: value for group_id, value in
                           zip(x['groups'], model_prediction)}
        return prediction_dict

    def _build_prediction_df(self, current_state):
        new_data = []

        for group_name, group_state in current_state.env_state.items():
            for history_idx, history_value in enumerate(group_state.history[1:], start=1):
                new_data = self._add_sample_to_data(new_data, history_value, group_name, history_idx)

        first_env_state_group_name = next(iter(current_state.env_state))
        new_data = self._add_current_state_to_data(new_data, first_env_state_group_name, current_state)
        new_data = self._add_dummy_sample_to_data(new_data, first_env_state_group_name, current_state)

        prediction_df = pd.concat([self.init_prediction_df, pd.DataFrame.from_dict(new_data)], axis=0)
        for dt_column in self.config.get("DatetimeAdditionalColumns", []):
            prediction_df[dt_column] = prediction_df[dt_column].astype(str).astype("category")
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def _add_sample_to_data(self, new_data, value, series, idx_diff):
        data = {self.config.get("GroupKeyword"): series,
                self.config.get("ValueKeyword"): value,
                DATETIME_COLUMN: self.last_val_date + datetime.timedelta(
                    hours=idx_diff) if self.last_val_date else None,
                'time_idx': self.last_val_time_idx + idx_diff
                }

        dt_columns = self.config.get("DatetimeAdditionalColumns", [])
        add_dt_columns(data, dt_columns)
        new_data.append(data)
        return new_data

    def _add_dummy_sample_to_data(self, new_data, first_env_state_group_name, current_state):
        if not new_data:
            dummy_data = self.test_df[lambda x: x.time_idx == self.last_val_time_idx].to_dict('records')
        else:
            dummy_data = new_data[-self.num_series:]
        idx_diff = len(current_state.env_state[first_env_state_group_name].history) + 1

        for sample in dummy_data:
            series = sample[self.config.get("GroupKeyword")]
            value = sample[self.config.get("ValueKeyword")]
            new_data = self._add_sample_to_data(new_data, value, series, idx_diff)

        return new_data

    def _add_current_state_to_data(self, new_data, first_env_state_group_name, current_state):
        idx_diff = len(current_state.env_state[first_env_state_group_name].history)
        if idx_diff > 0:
            for group_name, group_state in current_state.env_state.items():
                new_data = self._add_sample_to_data(new_data,
                                                    group_state.value,
                                                    group_name,
                                                    idx_diff)
        return new_data

    def sample_from_prediction(self, model_prediction, group_id, chosen_quantile):
        sampled_prediction = {}
        quantile_idx_list = np.random.randint(low=1, high=self.num_quantiles - 1, size=self.num_series)
        quantile_idx_list[group_id] = chosen_quantile

        for group_id, (group_name, group_prediction) in enumerate(model_prediction.items()):
            sampled_prediction[group_name] = group_prediction[0][quantile_idx_list[group_id]]

        return sampled_prediction
