from abc import ABC
from pytorch_forecasting import TimeSeriesDataSet
from utils import load_pickle
import os
import pandas as pd
import numpy as np
from config import DATETIME_COLUMN


class DataBuilder(ABC):
    def __init__(self, config):
        self.config = config
        self.train_ratio = config.get("Train").get("TrainRatio")
        self.val_ratio = config.get("Train").get("ValRatio")
        self.enc_length = config.get("EncoderLength")
        self.prediction_length = config.get("PredictionLength")

    def build_data(self) -> pd.DataFrame():
        if self.config.get("ProcessedDataPath") and os.path.isfile(self.config.get("ProcessedDataPath")):
            data = load_pickle(self.config.get("ProcessedDataPath"))
        else:
            data = self.get_data()
            data = self.preprocess(data)
        return data

    def build_ts_data(self, df: pd.DataFrame(), parameters=None, type_="reg"):
        if not parameters:
            if type_ == "reg":
                ts_ds = self.define_regression_ts_ds(df)
            elif type_ == "class":
                ts_ds = self.define_classification_ts_ds(df)
            else:
                raise ValueError
            parameters = ts_ds.get_parameters()
        else:
            ts_ds = TimeSeriesDataSet.from_parameters(parameters, df)
        return ts_ds, parameters

    def get_data(self):
        raise NotImplementedError

    @staticmethod
    def preprocess(data):
        return data

    @staticmethod
    def add_time_idx_column(data):
        pass

    @staticmethod
    def define_classification_ts_ds(train_df):
        pass

    @staticmethod
    def define_regression_ts_ds(train_df):
        raise NotImplementedError

    def split_df(self, data: pd.DataFrame()):
        training_max_idx = int(data.time_idx.max() * self.train_ratio)

        test_start_idx = int(data.time_idx.max() * (self.train_ratio + self.val_ratio)) - (
                self.enc_length + self.prediction_length)
        test_df = data[lambda x: x.time_idx > test_start_idx]

        train_df = data[lambda x: x.time_idx <= training_max_idx]

        validation_start_idx = training_max_idx + 1 - (self.enc_length + self.prediction_length)
        validation_end_idx = test_start_idx + self.enc_length + self.prediction_length
        val_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]

        return train_df, val_df, test_df
