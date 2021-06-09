from DataBuilders.data_builder import DataBuilder
import pandas as pd
import numpy as np
from constants import Paths
import os
from data_utils import add_dt_columns


class Params:
    TOTAL_NUM_COLUMNS = 371
    COLUMN_FILTER_INDEX = 21
    FILENAME = os.path.join(Paths.ELECTRICITY, 'LD2011_2014.txt')
    CHUNKSIZE = 10000
    RAW_DF_COLUMN_PREFIX = 'MT_'
    PROCESSED_DF_COLUMN_NAMES = ['date', 'group', 'value']


class ElectrictyDataBuilder(DataBuilder):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_data():
        users_col_names = [Params.RAW_DF_COLUMN_PREFIX + str(i)
                           for i
                           in range(1, Params.TOTAL_NUM_COLUMNS)]
        filtered_user_col_names = users_col_names[:Params.COLUMN_FILTER_INDEX]
        col_names = [Params.PROCESSED_DF_COLUMN_NAMES[0]] + users_col_names
        col_dtypes = {col_name: np.float64 for col_name in users_col_names}
        df_reader = pd.read_table(Params.FILENAME,
                                  chunksize=Params.CHUNKSIZE,
                                  delimiter=';',
                                  header=0,
                                  names=col_names,
                                  usecols=[i for i in range(Params.COLUMN_FILTER_INDEX)],
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
        df = df[df.date >= "2012-01-01"]
        return df.reset_index(drop=True)

    @staticmethod
    def preprocess(data):
        data = add_dt_columns(data, ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend'])
        data = ElectrictyDataBuilder.add_time_idx_column(data)
        return data

    @staticmethod
    def add_time_idx_column(data):
        data['time_idx'] = data.date.dt.year * 12 * 31 * 24 + \
                           data.date.dt.month * 31 * 24 + \
                           data.date.dt.day * 24 + \
                           data.date.dt.hour
        data['time_idx'] -= data.time_idx.min()
        return data

    @staticmethod
    def define_ts_ds(train_df):
        pass


if __name__ == "__main__":
    data_helper = ElectrictyDataBuilder()
    data = data_helper.get_data()
    data = data_helper.preprocess(data)
    print(data.head(100))