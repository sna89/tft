import datetime
from datetime import timedelta
import numpy as np
import os
import pandas as pd
from data_utils import add_dt_columns
from pytorch_forecasting import TimeSeriesDataSet
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN
from utils import save_to_pickle
from pytorch_forecasting.data import NaNLabelEncoder


KEY_DELIMITER = '_'
KEY_COLUMN = 'key'
BOUNDS_PARAMETER_NAME_LIST = ['MinValue', 'MinWarnValue', 'MaxWarnValue', 'MaxValue']
PLANT_MODEL_ID = 2629
# MAX_ORDER_STEP_ID = 430000
MIN_DATETIME = datetime.datetime(day=1, month=4, year=2021)
QMP_FILTER_LIST = [6, 9, 11, 14, 19, 26, 53]
UNPLANNED_STOPPAGE_TYPE_ID = 4
MIN_STOPPAGE_DURATION = timedelta(minutes=15)


class StrausDataBuilder(DataBuilder):
    def __init__(self, config):
        super(StrausDataBuilder, self).__init__(config)
        self.bounds = {}

    def get_data(self):
        qmp_log_df, order_log_df, stoppage_log_df, stoppage_event_df = self.read_files()
        qmp_order_log_df = self.join_qmp_order_log_data(qmp_log_df, order_log_df)
        stoppage_log_event_df = self.join_stoppage_log_event_data(stoppage_log_df, stoppage_event_df)
        stoppage_log_event_df = self.process_joined_stoppage_data(stoppage_log_event_df)
        qmp_order_log_df = self.add_stoppage_ind_to_qmp_order_log_data(qmp_order_log_df, stoppage_log_event_df)
        self._create_key_column(qmp_order_log_df)
        for group_col in self.config.get("GroupColumns") + [KEY_COLUMN]:
            qmp_order_log_df[group_col] = qmp_order_log_df[group_col].astype(str).astype("category")
        # self._get_bounds(filename, data)
        qmp_order_log_df.drop_duplicates(subset=["QmpId", "PartId", "OrderStepId", 'TimeStmp'], inplace=True)
        return qmp_order_log_df

    def preprocess(self, data):
        data = self._add_dt_column(data)
        data = self._sort_data(data)
        data = self._assign_time_idx(data)
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data.sort_values(by=[KEY_COLUMN, DATETIME_COLUMN], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        save_to_pickle(data, self.config.get("ProcessedDataPath"))
        return data

    def read_files(self):
        filenames = os.listdir(self.config.get("Path"))

        qmp_log_data_list = []
        order_log_data_list = []
        stoppage_log_data_list = []
        stoppage_event_data_list = []

        for filename in filenames:
            file_path = os.path.join(self.config.get("Path"), filename)
            if 'QmpLog_data' in filename:
                df = self.read_qmp_log_file(file_path)
                qmp_log_data_list.append(df)
            elif 'StoppageLog_data' in filename:
                df = self.read_stoppage_log_file(file_path)
                stoppage_log_data_list.append(df)
            elif 'OrderStepLog_data' in filename:
                df = self.read_order_log_file(file_path)
                order_log_data_list.append(df)
            elif 'StoppageEvent_data' in filename:
                df = self.read_stoppage_event_file(file_path)
                stoppage_event_data_list.append(df)

        qmp_log_df = pd.concat(qmp_log_data_list, axis=0)
        order_log_df = pd.concat(order_log_data_list, axis=0)
        stoppage_log_df = pd.concat(stoppage_log_data_list, axis=0)
        stoppage_event_df = pd.concat(stoppage_event_data_list, axis=0)

        return qmp_log_df, order_log_df, stoppage_log_df, stoppage_event_df

    @staticmethod
    def read_qmp_log_file(file_path):
        raw_df = pd.read_csv(file_path)
        df = StrausDataBuilder.filter_by_plant_model_id(raw_df)
        df.drop(columns=['PlantModelId', 'SpValue'], inplace=True)
        df['TimeStmp'] = pd.to_datetime(df['TimeStmp'])
        df = StrausDataBuilder.filter_by_qmp(df)
        df = StrausDataBuilder.filter_qmp_df_by_dt(df)
        # df = StrausDataBuilder.agg_mean_qmp_by_qmp_order_step(df)
        return df

    @staticmethod
    def filter_by_qmp(df):
        df = df[df.QmpId.isin(QMP_FILTER_LIST)]
        return df

    @staticmethod
    def filter_qmp_df_by_dt(df):
        df = df[df.TimeStmp >= MIN_DATETIME]
        return df

    @staticmethod
    def filter_by_plant_model_id(df):
        df = df[(df.PlantModelId == PLANT_MODEL_ID)]
        return df

    @staticmethod
    def agg_mean_qmp_by_qmp_order_step(df, freq='10T'):
        dfs = []
        df[KEY_COLUMN] = df['OrderStepId'].astype(str) + KEY_DELIMITER + df['QmpId'].astype(str)
        for key in pd.unique(df[KEY_COLUMN]):
            sub_df = df[df[KEY_COLUMN] == key]
            agg_sub_df = sub_df[['TimeStmp', 'ActualValue']].set_index('TimeStmp').resample(freq).mean().dropna()
            agg_sub_df['QmpId'] = pd.unique(sub_df['QmpId'])[0]
            agg_sub_df['OrderStepId'] = pd.unique(sub_df['OrderStepId'])[0]
            agg_sub_df['TargetValue'] = pd.unique(sub_df['TargetValue'])[0]
            agg_sub_df['MinValue'] = pd.unique(sub_df['MinValue'])[0]
            agg_sub_df['MinWarnValue'] = pd.unique(sub_df['MinWarnValue'])[0]
            agg_sub_df['MaxWarnValue'] = pd.unique(sub_df['MaxWarnValue'])[0]
            agg_sub_df['MaxValue'] = pd.unique(sub_df['MaxValue'])[0]
            agg_sub_df['TimeStmp'] = agg_sub_df.index
            agg_sub_df.reset_index(drop=True, inplace=True)
            dfs.append(agg_sub_df)
        df = pd.concat(dfs, axis=0)
        return df

    @staticmethod
    def read_stoppage_log_file(file_path):
        df = pd.read_csv(file_path)
        df = df[df.PlantModelId == PLANT_MODEL_ID]
        df.drop(columns=['PlantModelId', 'BlamePlantModelId'], inplace=True)
        df['ActualStrTime'] = pd.to_datetime(df['ActualStrTime'])
        df['ActualEndTime'] = pd.to_datetime(df['ActualEndTime'])
        df = df[df.ActualStrTime >= MIN_DATETIME]
        return df

    @staticmethod
    def read_order_log_file(file_path):
        df = pd.read_csv(file_path)
        df = df[df.PlantModelId == PLANT_MODEL_ID]
        df.drop(columns=['PlantModelId', 'OrderStepIdentity', 'ActualSetupTime', 'SetupTime', 'UomId'],
                inplace=True)
        df['ActualStrTime'] = pd.to_datetime(df['ActualStrTime'])
        df['ActualEndTime'] = pd.to_datetime(df['ActualEndTime'])
        df = df[df.ActualStrTime >= MIN_DATETIME]
        return df

    @staticmethod
    def read_stoppage_event_file(file_path):
        df = pd.read_csv(file_path)
        df = df.fillna(0)
        df = df.astype({"StoppageType": int})
        # df = df[df.StoppageType == UNPLANNED_STOPPAGE_TYPE_ID]
        df.drop(columns=['StoppageEventName', 'StoppageTypeName', "StoppageType"],
                inplace=True)
        return df

    @staticmethod
    def add_stoppage_ind_to_qmp_order_log_data(qmp_order_log_df, stoppage_log_event_df):
        qmp_order_log_df['is_stoppage'] = 0

        for idx, stoppage_log_event in stoppage_log_event_df.iterrows():
            stoppage_start_time = stoppage_log_event['ActualStrTime']
            stoppage_end_time = stoppage_log_event['ActualEndTime']
            stoppage_qmp_df = qmp_order_log_df[(qmp_order_log_df.TimeStmp >= stoppage_start_time) &
                                               (qmp_order_log_df.TimeStmp <= stoppage_end_time)]
            qmp_order_log_df.loc[stoppage_qmp_df.index, 'is_stoppage'] = 1

        qmp_order_log_df['is_stoppage'] = qmp_order_log_df['is_stoppage'].astype(str).astype("category")
        return qmp_order_log_df

    @staticmethod
    def join_qmp_order_log_data(qmp_log_df, order_log_df):
        qmp_order_log_joined_df = qmp_log_df.set_index("OrderStepId").join(order_log_df.set_index("OrderStepId"),
                                                                           on="OrderStepId",
                                                                           how='right',
                                                                           lsuffix='_caller',
                                                                           rsuffix='_other')
        qmp_order_log_joined_df.drop(columns=["ActualStrTime", "ActualEndTime", "ActualQty"], inplace=True)
        qmp_order_log_joined_df['OrderStepId'] = qmp_order_log_joined_df.index
        qmp_order_log_joined_df.dropna(inplace=True)
        qmp_order_log_joined_df.sort_values(by="TimeStmp", inplace=True)
        qmp_order_log_joined_df.reset_index(drop=True, inplace=True)
        return qmp_order_log_joined_df

    @staticmethod
    def join_stoppage_log_event_data(stoppage_log_df, stoppage_event_df):
        joined_stoppage_log_event_df = stoppage_event_df.set_index("StoppageEventId").join(
            stoppage_log_df.set_index("StoppageEventId"),
            on="StoppageEventId",
            how="inner",
            lsuffix='_caller',
            rsuffix='_other'
        )
        joined_stoppage_log_event_df = joined_stoppage_log_event_df.sort_values(by='ActualStrTime')
        joined_stoppage_log_event_df.reset_index(drop=True, inplace=True)
        return joined_stoppage_log_event_df

    @staticmethod
    def process_joined_stoppage_data(df):
        df['StoppageDuration'] = df['ActualEndTime'] - \
                                 df['ActualStrTime']
        # df['TimeDiffBetweenStoppages'] = df.shift(-1)['ActualStrTime'] - \
        #                                  df['ActualEndTime']
        df = df[df['StoppageDuration'] > MIN_STOPPAGE_DURATION]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _filter_data(data):
        data = data[data.QmpId.isin([str(i) for i in range(1, 7)])]
        return data

    @staticmethod
    def _sort_data(data):
        data = data.sort_values(by=DATETIME_COLUMN)
        return data

    @staticmethod
    def _add_dt_column(data):
        data[DATETIME_COLUMN] = pd.to_datetime(data['TimeStmp'])
        return data

    @staticmethod
    def _create_key_column(data):
        data[KEY_COLUMN] = data['PartId'].astype(str) + KEY_DELIMITER + \
                           data['OrderStepId'].astype(int).astype(str) + KEY_DELIMITER + \
                           data['QmpId'].astype(int).astype(str)

    def _get_bounds(self, filename, data):
        if 'CI_QmpLog_data' in filename:
            keys_list = pd.unique(data[KEY_COLUMN])
            for key in keys_list:
                plant_model_id, order_step_id, qmp_id = key.split(KEY_DELIMITER)

                if plant_model_id not in self.bounds:
                    self.bounds[plant_model_id] = {}
                if order_step_id not in self.bounds[plant_model_id]:
                    self.bounds[plant_model_id][order_step_id] = {}
                if qmp_id not in self.bounds[plant_model_id][order_step_id]:
                    self.bounds[plant_model_id][order_step_id][qmp_id] = {}
                    key_sub_df = self.get_key_sub_df(data, key)
                    bounds_df = key_sub_df[BOUNDS_PARAMETER_NAME_LIST]

                    for bound_parameter in BOUNDS_PARAMETER_NAME_LIST:
                        unique_arr = pd.unique(bounds_df[bound_parameter])
                        self.bounds[plant_model_id][order_step_id][qmp_id][bound_parameter] = unique_arr[0]
                        if unique_arr.shape[0] > 1:
                            print(plant_model_id)
                            print(order_step_id)
                            print(qmp_id)
                            print(bound_parameter)
                            print(unique_arr)

    def _assign_time_idx(self, df):
        df['time_idx'] = None
        keys = pd.unique(df[KEY_COLUMN])
        for key in keys:
            sub_df = df[df[KEY_COLUMN] == key]
            time_index = pd.unique(sub_df[DATETIME_COLUMN])
            time_index_time_idx_mapping = dict(zip(pd.to_datetime(time_index), list(range(1, len(time_index) + 1))))
            df.loc[sub_df.index, 'time_idx'] = sub_df.apply(lambda x: self._get_time_idx(x, time_index_time_idx_mapping), axis=1)
        return df

    @staticmethod
    def _get_time_idx(row, time_index_time_idx_mapping):
        time_idx = time_index_time_idx_mapping[pd.to_datetime(row[DATETIME_COLUMN])]
        return time_idx

    @staticmethod
    def get_key_sub_df(data, key):
        plant_model_id, order_step_id, qmp_id = key.split(KEY_DELIMITER)
        key_sub_df = data[(data.PlantModelId == plant_model_id) &
                          (data.OrderStepId == order_step_id) &
                          (data.QmpId == qmp_id)]
        return key_sub_df

    def define_ts_ds(self, train_df):
        straus_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="is_stoppage",
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=[self.config.get("GroupKeyword")],
            static_reals=[],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            time_varying_known_reals=["time_idx", "TargetValue"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=False,
            allow_missings=True,
            categorical_encoders={self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True),
                                  **{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                                     self.config.get("DatetimeAdditionalColumns")}},
        )
        return straus_train_ts_ds