import datetime
from datetime import timedelta
import os
import pandas as pd
from data_utils import filter_df_by_date, add_dt_columns
from pytorch_forecasting import TimeSeriesDataSet
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN


class FishermanDataBuilder(DataBuilder):
    def __init__(self, config):
        super(FishermanDataBuilder, self).__init__(config)

    def get_data(self):
        dfs = []
        max_min_date = datetime.datetime(year=1900, month=1, day=1)
        min_max_date = datetime.datetime(year=2222, month=1, day=1)
        for filename in os.listdir(self.config.get("Path")):
            full_file_path = os.path.join(self.config.get("Path"), filename)
            df = pd.read_csv(full_file_path, usecols=['Type', 'Value', 'Time'])
            df = df[df.Type == 'internaltemp']
            df = df.drop(columns=['Type'], axis=1)
            df[DATETIME_COLUMN] = pd.to_datetime(df['Time'])

            min_date_df = df[DATETIME_COLUMN].min()
            if min_date_df > max_min_date:
                max_min_date = min_date_df

            max_date_df = df[DATETIME_COLUMN].max()
            if max_date_df < min_max_date:
                min_max_date = max_date_df

            df[self.config.get("GroupKeyword")] = filename.replace('Sensor ', '').replace('.csv', '')
            dfs.append(df)

        dfs = list(map(lambda dfx: self._preprocess_single_df_fisherman(dfx,
                                                                        max_min_date + timedelta(minutes=10),
                                                                        min_max_date - timedelta(minutes=10)
                                                                        ),
                       dfs)
                   )
        data = pd.concat(dfs, axis=0)
        data.reset_index(inplace=True, drop=True)
        return data

    def _preprocess_single_df_fisherman(self, data, min_date, max_date):
        data = filter_df_by_date(data, min_date, max_date)
        data.drop_duplicates(inplace=True)
        sensor = data[self.config.get("GroupKeyword")].iloc[0]
        if self.config.get("Resample") == "True":
            data = data.set_index(DATETIME_COLUMN).resample('3H').mean()
            data[DATETIME_COLUMN] = data.index
        data.fillna(method='bfill', inplace=True)
        data[self.config.get("GroupKeyword")] = sensor
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data.reset_index(inplace=True, drop=True)
        data['time_idx'] = data.apply(lambda x: FishermanDataBuilder.set_row_time_idx(x), axis=1)
        return data

    def define_ts_ds(self, train_df):
        fisherman_train_ts_ds = TimeSeriesDataSet(
            train_df,
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
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            # target_normalizer=GroupNormalizer(
            #     groups=["Sensor"]
            # ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=False,
            allow_missings=True
        )
        return fisherman_train_ts_ds

    @staticmethod
    def set_row_time_idx(x):
        time_idx = x.name
        return time_idx
