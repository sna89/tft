from abc import ABC
from pytorch_forecasting import TimeSeriesDataSet
from utils import load_pickle
import os
import pandas as pd
import numpy as np
from config import DATETIME_COLUMN
from datetime import timedelta


class DataBuilder(ABC):
    def __init__(self, config):
        self.config = config
        self.train_ratio = config.get("Train").get("TrainRatio")
        self.val_ratio = config.get("Train").get("ValRatio")
        self.enc_length = config.get("EncoderLength")
        self.prediction_length = config.get("PredictionLength")

    def build_ts_data(self):
        # if self.config.get("ProcessedDataPath") and os.path.isfile(self.config.get("ProcessedDataPath")):
        #     data = load_pickle(self.config.get("ProcessedDataPath"))
        # else:
        data = self.get_data()
        data = self.preprocess(data)

        train_df, validation_df, test_df = self.split_dataframe_train_val_test(data)
        train_ts_ds = self.define_ts_ds(train_df)
        parameters = train_ts_ds.get_parameters()
        validation_ts_ds = TimeSeriesDataSet.from_parameters(parameters, validation_df)
        test_ts_ds = TimeSeriesDataSet.from_parameters(parameters, test_df)
        return train_df, validation_df, test_df, train_ts_ds, validation_ts_ds, test_ts_ds

    def get_data(self):
        raise NotImplementedError

    @staticmethod
    def preprocess(data):
        return data

    @staticmethod
    def add_time_idx_column(data):
        pass

    @staticmethod
    def define_ts_ds(train_df):
        raise NotImplementedError

    def split_dataframe_train_val_test(self, data, by=DATETIME_COLUMN, method="random"):
        assert by in ["time_idx", DATETIME_COLUMN], "by parameter in split_dataframe_train_val_test must be either: " \
                                                    "time_idx, {}".format(DATETIME_COLUMN)
        assert method in ["ratio", "random"], "method parameter in split_dataframe_train_val_test must be either: " \
                                              "ratio, random"

        if method == "ratio" and by == "time_idx":
            training_max_idx = int(data.time_idx.max() * self.train_ratio)

            test_start_idx = int(data.time_idx.max() * (self.train_ratio + self.val_ratio)) - (
                    self.enc_length + self.prediction_length)
            test_df = data[lambda x: x.time_idx > test_start_idx]

            train_df = data[lambda x: x.time_idx <= training_max_idx]

            validation_start_idx = training_max_idx + 1 - (self.enc_length + self.prediction_length)
            validation_end_idx = test_start_idx + self.enc_length + self.prediction_length
            val_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]

        elif method == "random" and by == DATETIME_COLUMN:
            np.random.seed(42)

            train_df_list = []
            val_df_list = []

            date_index = pd.DatetimeIndex(data[DATETIME_COLUMN])
            date_index_total_time = date_index.max() - date_index.min()
            test_time_start_dt = date_index.min() + date_index_total_time * (self.train_ratio + self.val_ratio)

            test_df = data[lambda x: x[DATETIME_COLUMN] >= test_time_start_dt]
            data = data[lambda x: x[DATETIME_COLUMN] < test_time_start_dt]

            encoder_len = self.config.get("EncoderLength")
            prediction_len = self.config.get("PredictionLength")
            groups = pd.unique(data[self.config.get("GroupKeyword")])
            for i, group in enumerate(groups):
                sub_df = data[data[self.config.get("GroupKeyword")] == group]
                max_time_idx = sub_df.time_idx.max()
                if max_time_idx <= encoder_len + prediction_len:
                    continue

                random_time_idx_list = list(np.random.choice(np.arange(encoder_len, max_time_idx - prediction_len),
                                                             size=(10, 3)))
                for time_idx in random_time_idx_list:
                    train_sub_df = sub_df[(sub_df.time_idx >= time_idx[0] - encoder_len)
                                          & (sub_df.time_idx <= time_idx[0] + prediction_len)]
                    val_sub_df = sub_df[(sub_df.time_idx >= time_idx[1] - encoder_len)
                                        & (sub_df.time_idx <= time_idx[1] + prediction_len)]

                    train_df_list.append(train_sub_df)
                    val_df_list.append(val_sub_df)

            train_df = pd.concat(train_df_list, axis=0)
            val_df = pd.concat(val_df_list, axis=0)

        train_df = train_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']).\
            sort_values('time_idx').\
            reset_index(drop=True)
        val_df = val_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']).\
            sort_values('time_idx').\
            reset_index(drop=True)
        test_df = test_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']).\
            sort_values('time_idx').\
            reset_index(drop=True)

        return train_df, val_df, test_df
