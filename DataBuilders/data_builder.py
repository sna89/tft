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
        self.train_ratio = config.get("TrainRatio")
        self.val_ratio = config.get("ValRatio")
        self.enc_length = config.get("EncoderLength")
        self.prediction_length = config.get("PredictionLength")

    def build_ts_data(self):
        if self.config.get("load_data_path"):
            if os.path.isfile(self.config.get("load_data_path")):
                data = load_pickle(self.config.get("load_data_path"))
            else:
                raise ValueError(self.config.get("load_data_path") + " does not exists")
        else:
            data = self.get_data()
            data = self.preprocess(data)
            if self.config.get("save_data_path"):
                filetype = self.config.get("save_data_path").split('.')[-1]
                if filetype == "csv":
                    data.to_csv(self.config.get("save_data_path"))
                elif filetype == "pkl":
                    data.to_pickle(self.config.get("save_data_path"))
                else:
                    raise ValueError("File path currently not supported")

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

    def split_dataframe_train_val_test(self, data):
        train_ratio = self.config.get("DataLoader").get("TrainRatio")
        val_ratio = self.config.get("DataLoader").get("ValRatio")
        enc_length = self.config.get("Data").get("EncoderLength")
        pred_length = self.config.get("Data").get("PredictionLength")

        training_max_idx = int(data.time_idx.max() * train_ratio)

        test_start_idx = int(data.time_idx.max() * (train_ratio + val_ratio)) - (
                enc_length + pred_length)
        test_df = data[lambda x: x.time_idx >= test_start_idx]

        train_df = data[lambda x: x.time_idx <= training_max_idx]

        validation_start_idx = training_max_idx + 1 - (enc_length + pred_length)
        validation_end_idx = test_start_idx + enc_length + pred_length
        val_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]

        return train_df, val_df, test_df
