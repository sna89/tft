from DataBuilders.data_builder import DataBuilder
import pandas as pd
import numpy as np
from constants import Paths
import os
from data_utils import add_dt_columns
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder


class Params:
    TOTAL_NUM_COLUMNS = 371
    NUM_GROUPS = 11
    FILENAME = os.path.join(Paths.ELECTRICITY, 'LD2011_2014.txt')
    CHUNKSIZE = 10000
    RAW_DF_COLUMN_PREFIX = 'MT_'
    PROCESSED_DF_COLUMN_NAMES = ['date', 'group', 'value']
    ENCODER_LENGTH = 168
    PREDICTION_LENGTH = 24
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    START_DATE = "2012-01-01"
    END_DATE = "2013-01-01"


class ElectricityDataBuilder(DataBuilder):
    def __init__(self, train_ratio, val_ratio, enc_length, prediction_length):
        super(ElectricityDataBuilder, self).__init__(train_ratio, val_ratio, enc_length, prediction_length)

    @staticmethod
    def get_data():
        users_col_names = [Params.RAW_DF_COLUMN_PREFIX + str(i)
                           for i
                           in range(1, Params.TOTAL_NUM_COLUMNS)]
        filtered_user_col_names = users_col_names[:Params.NUM_GROUPS]
        col_names = [Params.PROCESSED_DF_COLUMN_NAMES[0]] + users_col_names
        col_dtypes = {col_name: np.float64 for col_name in users_col_names}
        df_reader = pd.read_table(Params.FILENAME,
                                  chunksize=Params.CHUNKSIZE,
                                  delimiter=';',
                                  header=0,
                                  names=col_names,
                                  usecols=[i for i in range(Params.NUM_GROUPS)],
                                  parse_dates=[0],
                                  decimal=',',
                                  dtype=col_dtypes,
                                  index_col=0)
        dfs = list()
        for df_batch_raw in df_reader:
            df_batch_resample = df_batch_raw.resample('1H').mean()
            df_batch = df_batch_resample.stack().reset_index()
            df_batch.rename(columns={'level_1': Params.PROCESSED_DF_COLUMN_NAMES[1],
                                     0: Params.PROCESSED_DF_COLUMN_NAMES[2]},
                            inplace=True)
            df_batch = df_batch[df_batch.group.isin(filtered_user_col_names)].reset_index(drop=True)
            dfs.append(df_batch)
        df = pd.concat(dfs, axis=0)
        df = df.drop_duplicates(subset=Params.PROCESSED_DF_COLUMN_NAMES[:2])
        df = df[(df.date >= Params.START_DATE) & (df.date < Params.END_DATE)]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def preprocess(data):
        data = add_dt_columns(data, ['hour', 'day_of_week', 'day_of_month'])
        data = ElectricityDataBuilder.add_time_idx_column(data)
        return data

    @staticmethod
    def add_time_idx_column(data):
        dt_index = pd.DatetimeIndex(data.date.unique())
        date_time_idx_map = dict(zip(dt_index, range(1, len(dt_index) + 1)))
        data['time_idx'] = data.apply(lambda row: date_time_idx_map[row.date], axis=1)
        return data

    def define_ts_ds(self, train_df):
        electricity_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            group_ids=[Params.PROCESSED_DF_COLUMN_NAMES[1]],
            target=Params.PROCESSED_DF_COLUMN_NAMES[2],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            static_categoricals=[Params.PROCESSED_DF_COLUMN_NAMES[1]],
            time_varying_known_categoricals=['hour', 'day_of_week', 'day_of_month'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[Params.PROCESSED_DF_COLUMN_NAMES[2]],
            # target_normalizer=GroupNormalizer(
            #     groups=[Params.PROCESSED_DF_COLUMN_NAMES[1]]
            #     # ,transformation="softplus"
            # ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        return electricity_train_ts_ds


if __name__ == "__main__":
    data_helper = ElectricityDataBuilder(Params.TRAIN_RATIO,
                                         Params.VAL_RATIO,
                                         Params.ENCODER_LENGTH,
                                         Params.PREDICTION_LENGTH)
    train_df, validation_df, test_df, train_ts_ds, validation_ts_ds, test_ts_ds = data_helper.build_ts_data()