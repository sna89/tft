from abc import ABC
from pytorch_forecasting import TimeSeriesDataSet
from utils import load_pickle
import os
import pandas as pd
import numpy as np
from config import DATETIME_COLUMN, \
    REGRESSION_TASK_TYPE, \
    CLASSIFICATION_TASK_TYPE,\
    COMBINED_TASK_TYPE, \
    ROLLOUT_TASK_TYPE


class DataBuilder(ABC):
    def __init__(self, config):
        self.config = config
        self.train_ratio = config.get("Train").get("TrainRatio")
        self.val_ratio = config.get("Train").get("ValRatio")
        self.enc_length = config.get("EncoderLength")
        self.prediction_length = config.get("PredictionLength")

    def build_ts_data(self, df: pd.DataFrame(), parameters, task_type):
        if not parameters:
            if task_type in [REGRESSION_TASK_TYPE, ROLLOUT_TASK_TYPE]:
                ts_ds = self.define_regression_ts_ds(df)
            elif task_type == CLASSIFICATION_TASK_TYPE:
                ts_ds = self.define_classification_ts_ds(df)
            elif task_type == COMBINED_TASK_TYPE:
                ts_ds = self.define_combined_ts_ds(df)
            else:
                raise ValueError
            parameters = ts_ds.get_parameters()
        else:
            ts_ds = TimeSeriesDataSet.from_parameters(parameters, df)
        return ts_ds, parameters

    def build_data(self):
        raise NotImplementedError

    def preprocess(self, data):
        raise NotImplementedError

    @staticmethod
    def add_time_idx_column(data):
        pass

    @staticmethod
    def define_classification_ts_ds(train_df):
        pass

    @staticmethod
    def define_combined_ts_ds(train_df):
        pass

    @staticmethod
    def define_regression_ts_ds(train_df):
        raise NotImplementedError

    def split_train_val_test(self, data: pd.DataFrame()):
        num_samples = (data.time_idx.max() + 1)

        training_max_idx = int(num_samples * self.train_ratio)

        test_start_idx = int(num_samples * (self.train_ratio + self.val_ratio)) - (
                self.enc_length + self.prediction_length)
        test_df = data[lambda x: x.time_idx > test_start_idx]

        train_df = data[lambda x: x.time_idx <= training_max_idx]

        validation_start_idx = training_max_idx + 1 - (self.enc_length + self.prediction_length)
        validation_end_idx = test_start_idx + self.enc_length + self.prediction_length
        val_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]

        return train_df, val_df, test_df
