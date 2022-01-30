import datetime

import torch

from EnvCommon.env_thts_common import get_last_val_time_idx, get_last_val_date, get_num_series
from config import DATETIME_COLUMN, REGRESSION_TASK_TYPE
from data_utils import get_group_id_group_name_mapping, add_dt_columns, get_dataloader, reverse_key_value_mapping
from config import get_num_quantiles
from utils import get_prediction_mode
import numpy as np
import pandas as pd
import time


class Predictor:
    def __init__(self, config, forecasting_model, test_df, test_ts_ds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.forecasting_model = forecasting_model.to(device)
        self.test_df = test_df
        self.test_ts_ds = test_ts_ds

        self.last_val_time_idx = get_last_val_time_idx(self.config, self.test_df)
        self.last_val_date = get_last_val_date(self.test_df, self.last_val_time_idx)
        self.init_prediction_df = None

        self.group_id_group_name_mapping = get_group_id_group_name_mapping(self.config, self.test_ts_ds)
        self.group_name_group_id_mapping = reverse_key_value_mapping(self.group_id_group_name_mapping)

        self.num_quantiles = get_num_quantiles()
        self.num_series = get_num_series(self.config, self.test_df)

    def predict(self, current_state, iteration=0):
        # start = time.time()
        prediction_df = self._build_prediction_df(current_state, iteration)
        # end = time.time()
        # run_time = end - start
        # print(run_time)
        prediction_mode = get_prediction_mode()
        model_prediction, x = self.forecasting_model.predict(prediction_df, mode=prediction_mode, return_x=True)

        if (isinstance(model_prediction, dict) and "prediction" in model_prediction) or \
                (isinstance(model_prediction, tuple) and "prediction" in model_prediction.keys()):
            model_prediction = model_prediction["prediction"]

        prediction_dict = {self.group_id_group_name_mapping[group_id.item()]: value for group_id, value in
                           zip(x['groups'], model_prediction)}
        return prediction_dict

    def _build_prediction_df(self, current_state, iteration):
        self.init_prediction_df = self.build_initial_prediction_df(iteration)
        new_prediction_df = self.build_new_prediction_df(current_state, iteration)

        prediction_df = pd.concat([self.init_prediction_df, new_prediction_df], axis=0)
        for dt_column in self.config.get("DatetimeAdditionalColumns", []):
            prediction_df[dt_column] = prediction_df[dt_column].astype(str).astype("category")
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df

    def build_initial_prediction_df(self, iteration):
        return self.test_df[self.test_df.time_idx <= self.last_val_time_idx + iteration]

    def build_new_prediction_df(self, current_state, iteration):
        new_data = []

        for group_name, group_state in current_state.env_state.items():
            for history_idx, history_value in enumerate(group_state.history[1:], start=1):
                new_data = self._add_sample_to_data(new_data, history_value, group_name, history_idx + iteration)

        first_env_state_group_name = next(iter(current_state.env_state))
        new_data = self._add_current_state_to_data(new_data, first_env_state_group_name, current_state, iteration)
        new_data = self._add_dummy_sample_to_data(new_data, first_env_state_group_name, current_state, iteration)

        df = pd.DataFrame.from_dict(new_data)
        return df

    def _add_sample_to_data(self, new_data, value, group, idx_diff):
        data = {self.config.get("GroupKeyword"): group,
                self.config.get("ValueKeyword"): value,
                DATETIME_COLUMN: self.last_val_date + datetime.timedelta(
                    hours=idx_diff) if self.last_val_date else None,
                'time_idx': self.last_val_time_idx + idx_diff
                }

        dt_columns = self.config.get("DatetimeAdditionalColumns", [])
        add_dt_columns(data, dt_columns)
        new_data.append(data)
        return new_data

    def _add_dummy_sample_to_data(self, new_data, first_env_state_group_name, current_state, iteration):
        if not new_data:
            dummy_data = self.test_df[lambda x: x.time_idx == self.last_val_time_idx + iteration].to_dict('records')
        else:
            dummy_data = new_data[-self.num_series:]

        for decoder_step in range(self.config.get("PredictionLength")):
            idx_diff = len(current_state.env_state[first_env_state_group_name].history) + 1 + decoder_step + iteration
            for sample in dummy_data:
                group = sample[self.config.get("GroupKeyword")]
                value = sample[self.config.get("ValueKeyword")]
                new_data = self._add_sample_to_data(new_data, value, group, idx_diff)

        return new_data

    def _add_current_state_to_data(self, new_data, first_env_state_group_name, current_state, iteration):
        idx_diff = len(current_state.env_state[first_env_state_group_name].history)
        if idx_diff > 0:
            for group_name, group_state in current_state.env_state.items():
                new_data = self._add_sample_to_data(new_data,
                                                    group_state.value,
                                                    group_name,
                                                    idx_diff + iteration)
        return new_data

    def sample_from_prediction(self, model_prediction, group_id, chosen_quantile):
        sampled_prediction = {}
        quantile_idx_list = np.random.randint(low=1, high=self.num_quantiles - 1, size=self.num_series)
        quantile_idx_list[group_id] = chosen_quantile

        for _, (group_name, group_prediction) in enumerate(model_prediction.items()):
            group_id = self.group_name_group_id_mapping[group_name]
            sampled_prediction[group_name] = group_prediction[0][quantile_idx_list[group_id]]

        return sampled_prediction
