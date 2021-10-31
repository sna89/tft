from DataBuilders.data_builder import DataBuilder
import pandas as pd
import numpy as np
import os
from data_utils import add_dt_columns
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder


class ElectricityDataBuilder(DataBuilder):
    def __init__(self, train_ratio,
                 val_ratio,
                 enc_length,
                 prediction_length,
                 file_path,
                 num_groups,
                 processed_df_column_names,
                 start_date,
                 end_date):
        super(ElectricityDataBuilder, self).__init__(train_ratio,
                                                     val_ratio,
                                                     enc_length,
                                                     prediction_length)
        self.file_path = file_path
        self.raw_df_column_prefix = "MT_"
        self.total_num_columns = 371
        self.num_groups = num_groups
        self.processed_df_column_names = processed_df_column_names
        self.start_date = start_date
        self.end_date = end_date

    def build_data(self):
        users_col_names = [self.raw_df_column_prefix + str(i)
                           for i
                           in range(1, self.total_num_columns)]
        filtered_user_col_names = users_col_names[:self.num_groups]
        col_names = [self.processed_df_column_names[0]] + users_col_names
        col_dtypes = {col_name: np.float64 for col_name in users_col_names}
        df_reader = pd.read_table(self.file_path,
                                  chunksize=10000,
                                  delimiter=';',
                                  header=0,
                                  names=col_names,
                                  usecols=[i for i in range(self.num_groups)],
                                  parse_dates=[0],
                                  decimal=',',
                                  dtype=col_dtypes,
                                  index_col=0)
        dfs = list()
        for df_batch_raw in df_reader:
            df_batch_resample = df_batch_raw.resample('1H').mean()
            df_batch = df_batch_resample.stack().reset_index()
            df_batch.rename(columns={'level_1': self.processed_df_column_names[1],
                                     0: self.processed_df_column_names[2]},
                            inplace=True)
            df_batch = df_batch[df_batch.group.isin(filtered_user_col_names)].reset_index(drop=True)
            dfs.append(df_batch)
        df = pd.concat(dfs, axis=0)
        df = df.drop_duplicates(subset=self.processed_df_column_names[:2])
        df = df[(df.date >= self.start_date) & (df.date < self.end_date)]
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

    def define_regression_ts_ds(self, train_df):
        electricity_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            group_ids=[self.processed_df_column_names[1]],
            target=self.processed_df_column_names[2],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            static_categoricals=[self.processed_df_column_names[1]],
            time_varying_known_categoricals=['hour', 'day_of_week', 'day_of_month'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[self.processed_df_column_names[2]],
            # target_normalizer=GroupNormalizer(
            #     groups=[Params.PROCESSED_DF_COLUMN_NAMES[1]]
            #     # ,transformation="softplus"
            # ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        return electricity_train_ts_ds
