from abc import ABC
from pytorch_forecasting import TimeSeriesDataSet


class DataBuilder(ABC):
    def __init__(self, train_ratio, val_ratio, enc_length, prediction_length):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.enc_length = enc_length
        self.prediction_length = prediction_length

    def build_ts_data(self):
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
    def preprocess():
        pass

    @staticmethod
    def add_time_idx_column(data):
        pass

    @staticmethod
    def define_ts_ds(train_df):
        raise NotImplementedError

    def split_dataframe_train_val_test(self, data):
        training_max_idx = int(data.time_idx.max() * self.train_ratio)
        train_df = data[lambda x: x.time_idx <= training_max_idx]

        test_start_idx = int(data.time_idx.max() * (self.train_ratio + self.val_ratio)) - (
                self.enc_length + self.prediction_length)
        test_df = data[lambda x: x.time_idx > test_start_idx]

        validation_start_idx = training_max_idx + 1 - (self.enc_length + self.prediction_length)
        validation_end_idx = test_start_idx + self.enc_length + self.prediction_length
        validation_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]
        return train_df, validation_df, test_df