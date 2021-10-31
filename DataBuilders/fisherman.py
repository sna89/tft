import datetime
from datetime import timedelta
import os
import pandas as pd
from data_utils import filter_df_by_date, add_dt_columns, assign_time_idx, add_future_exceed
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN


class FishermanDataBuilder(DataBuilder):
    def __init__(self, config):
        super(FishermanDataBuilder, self).__init__(config)

    def build_data(self):
        dfs = []
        for filename in os.listdir(self.config.get("Path")):
            full_file_path = os.path.join(self.config.get("Path"), filename)
            df = self._read_single_file(filename, full_file_path)
            dfs.append(df)

        max_min_date = max([df[DATETIME_COLUMN].min() for df in dfs])
        min_max_date = min([df[DATETIME_COLUMN].max() for df in dfs])
        dfs = list(map(lambda dfx: self._preprocess_single_df_fisherman(dfx,
                                                                        max_min_date,
                                                                        min_max_date
                                                                        ),
                       dfs)
                   )

        data = pd.concat(dfs, axis=0)
        data.reset_index(inplace=True, drop=True)
        return data

    def _read_single_file(self, filename, path):
        df = pd.read_csv(path, usecols=['Type', 'Value', 'Time'])
        df = df[df['Type'] == "internaltemp"]
        df.drop(columns=["Type"], inplace=True)
        df[DATETIME_COLUMN] = pd.to_datetime(df['Time'])
        df[self.config.get("GroupKeyword")] = filename.replace('Sensor ', '').replace('.csv', '')
        return df

    def _preprocess_single_df_fisherman(self, data, min_date, max_date):
        data = filter_df_by_date(data, min_date, max_date)
        data.drop_duplicates(inplace=True)
        sensor = data[self.config.get("GroupKeyword")].iloc[0]
        if self.config.get("Resample"):
            data = data.set_index(DATETIME_COLUMN).resample('1h').mean()
            data[DATETIME_COLUMN] = data.index
        data.fillna(method='bfill', inplace=True)
        data[self.config.get("GroupKeyword")] = sensor
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data.reset_index(inplace=True, drop=True)
        data = assign_time_idx(data, DATETIME_COLUMN)
        data = add_future_exceed(self.config, data, sensor)
        return data

    def define_regression_ts_ds(self, df):
        fisherman_ts_ds = TimeSeriesDataSet(
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
            time_varying_unknown_categoricals=[self.config.get("ExceptionKeyword")],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={"day_of_month": NaNLabelEncoder(add_nan=True)},
        )
        return fisherman_ts_ds

    def define_classification_ts_ds(self, train_exception_df):
        fisherman_train_exception_ts_ds = TimeSeriesDataSet(
            train_exception_df,
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
            allow_missing_timesteps=True
        )
        return fisherman_train_exception_ts_ds

    def preprocess(self, data):
        return data