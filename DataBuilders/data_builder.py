from abc import ABC
from pytorch_forecasting import TimeSeriesDataSet
from constants import DataConst, DataSetRatio


class DataBuilder(ABC):
    def __init__(self):
        pass

    @staticmethod
    def build_ts_data():
        data = DataBuilder.get_data()
        data = DataBuilder.preprocess(data)
        train_df, validation_df, test_df = DataBuilder.split_dataframe_train_val_test(data)
        train_ts_ds = DataBuilder.define_ts_ds(train_df)
        parameters = train_ts_ds.get_parameters()
        validation_ts_ds = TimeSeriesDataSet.from_parameters(parameters, validation_df)
        test_ts_ds = TimeSeriesDataSet.from_parameters(parameters, test_df)
        return train_df, validation_df, test_df, train_ts_ds, validation_ts_ds, test_ts_ds

    @staticmethod
    def get_data():
        raise NotImplementedError

    @staticmethod
    def preprocess():
        pass

    @staticmethod()
    def add_time_idx_column(data):
        pass

    @staticmethod
    def define_ts_ds(train_df):
        raise NotImplementedError

    @staticmethod
    def split_dataframe_train_val_test(data):
        training_max_idx = int(data.time_idx.max() * DataSetRatio.TRAIN)
        train_df = data[lambda x: x.time_idx <= training_max_idx]

        test_start_idx = int(data.time_idx.max() * (DataSetRatio.TRAIN + DataSetRatio.VAL)) - (
                DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
        test_df = data[lambda x: x.time_idx > test_start_idx]

        validation_start_idx = training_max_idx + 1 - (DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
        validation_end_idx = test_start_idx + DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH
        validation_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]
        return train_df, validation_df, test_df