from datetime import timedelta
import os
import pandas as pd
from data_utils import add_dt_columns, assign_time_idx, add_future_exceed
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN, KEY_DELIMITER
import numpy as np

MIN_TEMP_COL = 'Min Temp(C)'
MAX_TEMP_COL = 'Max Temp(C)'
MIN_RH_COL = 'Min RH'
MAX_RH_COL = 'Max RH'
SENSOR_COL = "Sensor"
INTERNAL_TEMP = "internaltemp"
INTERNAL_RH = "internalrh"
NUM_TIME_SERIES = 80
NUM_SAMPLES_IN_TIME_SERIES = 30 * 6


class Fisherman2DataBuilder(DataBuilder):
    def __init__(self, config):
        super(Fisherman2DataBuilder, self).__init__(config)

    def build_data(self):
        dfs = []
        for filename in os.listdir(self.config.get("Path")):
            full_file_path = os.path.join(self.config.get("Path"), filename)
            if 'metadata' not in filename:
                df = self._read_single_file(filename, full_file_path)
                dfs.append(df)
            # else:
            #     metadata_df = pd.read_csv(full_file_path, usecols=[SENSOR_COL,
            #                                                        MIN_TEMP_COL,
            #                                                        MAX_TEMP_COL])

        dfs = list(map(lambda dfx: self._preprocess_single_df_fisherman(dfx), dfs))
        data = pd.concat(dfs, axis=0)
        data.reset_index(inplace=True, drop=True)
        data = assign_time_idx(data, DATETIME_COLUMN)
        return data

    def preprocess(self, data):
        return data

    def _read_single_file(self, filename, path):
        df = pd.read_csv(path, usecols=['Type', 'Value', 'Time'])
        df = df[df['Type'] == "internaltemp"]
        df.drop(columns=["Type"], inplace=True)
        df[self.config.get("GroupKeyword")] = filename.replace('Sensor U', '').replace('.csv', '')
        return df

    def _get_keys(self, data):
        return pd.unique(data[self.config.get("GroupKeyword")])

    def _preprocess_single_df_fisherman(self, data):
        data.fillna(method='bfill', inplace=True)
        data[DATETIME_COLUMN] = pd.to_datetime(data['Time'])
        data = data.sort_values(by=DATETIME_COLUMN)

        if self.config.get("Resample") or self.config.get("SlidingWindow"):
            dfs = []
            keys = self._get_keys(data)
            for key in keys:
                sub_df = data[data[self.config.get("GroupKeyword")] == key]

                if self.config.get("Resample"):
                    sub_df = sub_df.set_index(DATETIME_COLUMN).resample('1h').mean()
                    sub_df[self.config.get("GroupKeyword")] = key
                    sub_df[DATETIME_COLUMN] = sub_df.index

                if self.config.get("SlidingWindow"):
                    sub_df[self.config.get("ValueKeyword")] = sub_df[self.config.get("ValueKeyword")]. \
                        rolling(window=self.config.get("SlidingWindowSamples")). \
                        mean()
                    sub_df.dropna(inplace=True)

                dfs.append(sub_df)
            data = pd.concat(dfs, axis=0)

        data = self._round_dt(data)
        data = data[[self.config.get("ValueKeyword"), self.config.get("GroupKeyword"), DATETIME_COLUMN]]
        data.drop_duplicates(inplace=True)
        add_dt_columns(data, dt_attributes=self.config.get("DatetimeAdditionalColumns"))
        return data

    # def _update_bounds(self, data, metadata):
    #     self.config['AnomalyConfig']['Fisherman2'] = {}
    #     keys = self._get_keys(data)
    #     for key in keys:
    #         bounds = metadata[metadata["Sensor"] == int(key)]
    #         lb = bounds[MIN_TEMP_COL].values[0]
    #         ub = bounds[MAX_TEMP_COL].values[0]
    #         self.config['AnomalyConfig']['Fisherman2'][key] = {'lb': lb,
    #                                                            'ub': ub}

    def define_regression_ts_ds(self, df):
        reg_ts_ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=[self.config.get("GroupKeyword")],
            static_reals=[],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True),
                                  **{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                                     self.config.get("DatetimeAdditionalColumns")}},
        )
        return reg_ts_ds

    def define_classification_ts_ds(self, exception_df):
        exception_ts_ds = TimeSeriesDataSet(
            exception_df,
            time_idx="time_idx",
            target=self.config.get("ExceptionKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=[self.config.get("GroupKeyword")],
            static_reals=[],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[self.config.get("ExceptionKeyword")],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True),
                                  **{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                                     self.config.get("DatetimeAdditionalColumns")}},

        )
        return exception_ts_ds

    def split_train_val_test(self, data: pd.DataFrame()):
        np.random.seed(42)

        train_dfs = []
        val_dfs = []
        test_dfs = []

        dt_index = pd.unique(data[DATETIME_COLUMN])
        idx = 0

        while idx < NUM_TIME_SERIES:
            train_dt_start, train_dt_end, val_dt_end, test_dt_end = self._get_train_val_test_times(dt_index)

            if test_dt_end in dt_index:
                dt_index = self._remove_from_dt_index(dt_index, train_dt_start, test_dt_end)

                train_df = self._get_sliced_df(data, train_dt_start, train_dt_end)
                train_df = self._update_time_series_key(train_df, idx, "tr")

                val_df = self._get_sliced_df(data, train_dt_end, val_dt_end)
                val_df = self._update_time_series_key(val_df, idx, "val")

                test_df = self._get_sliced_df(data, val_dt_end, test_dt_end)
                test_df = self._update_time_series_key(test_df, idx, "te")

                train_dfs.append(train_df)
                val_dfs.append(val_df)
                test_dfs.append(test_df)

                idx += 1

        train_df = pd.concat(train_dfs, axis=0)
        val_df = pd.concat(val_dfs, axis=0)
        test_df = pd.concat(test_dfs, axis=0)

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, val_df, test_df

    def _round_dt(self, data):
        data[DATETIME_COLUMN] -= np.array(data[DATETIME_COLUMN].dt.minute % 10, dtype='<m8[m]')
        data[DATETIME_COLUMN] -= np.array(data[DATETIME_COLUMN].dt.second % 60, dtype='<m8[s]')
        data.drop_duplicates(subset=[self.config.get("GroupKeyword"), DATETIME_COLUMN], inplace=True)
        data[DATETIME_COLUMN] = pd.to_datetime(pd.unique(data[DATETIME_COLUMN]))
        return data

    @staticmethod
    def _get_train_val_test_times(dt_index):
        train_dt_start = pd.to_datetime(np.random.choice(dt_index))
        train_dt_end = train_dt_start + timedelta(days=1, hours=6)
        val_dt_end = train_dt_end + timedelta(hours=6)
        test_dt_end = val_dt_end + timedelta(hours=6)
        return train_dt_start, train_dt_end, val_dt_end, test_dt_end

    @staticmethod
    def _remove_from_dt_index(dt_index, start_dt, end_dt):
        s = np.where(dt_index == start_dt)
        e = np.where(dt_index == end_dt)
        dt_index = np.delete(dt_index, np.s_[s[0][0]:e[0][0] + 1:1])
        return dt_index

    @staticmethod
    def _get_sliced_df(data, start_dt, end_dt):
        df = data[(data[DATETIME_COLUMN] >= start_dt) & (data[DATETIME_COLUMN] <= end_dt)]
        return df

    def _update_time_series_key(self, df, idx, type_="tr"):
        df[self.config.get("GroupKeyword")] = df[self.config.get("GroupKeyword")] + \
                                              KEY_DELIMITER + \
                                              str(idx) + \
                                              KEY_DELIMITER + \
                                              type_
        return df

    def _add_time_idx(self, data):
        dfs = []
        keys = self._get_keys(data)
        for key in keys:
            sub_df = data[data[self.config.get("GroupKeyword")] == key]
            sub_df = assign_time_idx(sub_df, DATETIME_COLUMN)
            dfs.append(sub_df)
        df = pd.concat(dfs, axis=0)
        return df


