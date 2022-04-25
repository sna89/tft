import datetime
from datetime import timedelta
import numpy as np
import os
import pandas as pd
from Utils.data_utils import add_dt_columns, assign_time_idx, add_bounds_to_config
from pytorch_forecasting import TimeSeriesDataSet
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN, KEY_DELIMITER
from pytorch_forecasting.data import NaNLabelEncoder

KEY_COLUMN = 'key'
BOUNDS_PARAMETER_NAME_LIST = ['MinValue', 'MinWarnValue', 'MaxWarnValue', 'MaxValue']
PLANT_MODEL_ID = 2629
# MAX_ORDER_STEP_ID = 430000
MIN_DATETIME = datetime.datetime(day=1, month=9, year=2021)
MAX_DATETIME = datetime.datetime(day=1, month=3, year=2022)
QMP_FILTER_LIST = [4, 6, 9, 11, 14, 18, 19, 26, 53]
QMP_ID_NIGHT_MODE = 19
NIGHT_MODE_THRESHOLD_VALUE = 35
UNPLANNED_STOPPAGE_TYPE_ID = 4
MIN_STOPPAGE_DURATION = timedelta(minutes=15)


class StrausDataBuilder(DataBuilder):
    def __init__(self, config):
        super(StrausDataBuilder, self).__init__(config)
        self.bounds = {}

    def build_data(self):
        qmp_log_df, order_log_df, stoppage_log_df, stoppage_event_df, temp_index_df = self.read_files()
        qmp_log_df = self.filter_night_mode(qmp_log_df)
        qmp_order_log_df = self.join_qmp_order_log_data(qmp_log_df, order_log_df)
        stoppage_log_event_df = self.join_stoppage_log_event_data(stoppage_log_df, stoppage_event_df)
        stoppage_log_event_df = self.process_joined_stoppage_data(stoppage_log_event_df)
        qmp_order_log_df = self.add_stoppage_ind_to_qmp_order_log_data(qmp_order_log_df, stoppage_log_event_df)
        # qmp_order_log_df = self.add_temp_index_to_qmp_order_log_data(qmp_order_log_df, temp_index_df)
        self._create_key_column(qmp_order_log_df)
        for group_col in self.config.get("GroupColumns") + [KEY_COLUMN]:
            qmp_order_log_df[group_col] = qmp_order_log_df[group_col].astype(str).astype("category")
        # self._get_bounds(filename, data)
        qmp_order_log_df.drop_duplicates(subset=["QmpId", "PartId", "OrderStepId", DATETIME_COLUMN], inplace=True)
        return qmp_order_log_df

    def preprocess(self, data):
        data = assign_time_idx(data, DATETIME_COLUMN)
        data = add_dt_columns(data, self.config.get("DatetimeAdditionalColumns"))
        data.sort_values(by=[DATETIME_COLUMN, KEY_COLUMN], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        # save_to_pickle(data, self.config.get("ProcessedDataPath"))
        return data

    def fill_missing_data(self, data):
        pass

    def read_files(self):
        filenames = os.listdir(self.config.get("Path"))

        qmp_log_data_list = []
        order_log_data_list = []
        stoppage_log_data_list = []
        stoppage_event_data_list = []
        temp_index_df = pd.DataFrame()

        for filename in filenames:
            file_path = os.path.join(self.config.get("Path"), filename)
            if 'QmpLog_data' in filename:
                df = self.read_qmp_log_file(file_path)
                df = self.filter_qmp_df(df)
                qmp_log_data_list.append(df)

            elif 'StoppageLog_data' in filename:
                df = self.read_stoppage_log_file(file_path)
                df = self.filter_stoppage_df(df)
                stoppage_log_data_list.append(df)

            elif 'OrderStepLog_data' in filename:
                df = self.read_order_log_file(file_path)
                df = self.filter_order_log_df(df)
                order_log_data_list.append(df)

            elif 'StoppageEvent_data' in filename:
                df = self.read_stoppage_event_file(file_path)
                df = self.filter_stoppage_event_df(df)
                stoppage_event_data_list.append(df)

            elif "TempIndex" in filename:
                temp_index_df = self.read_temp_index_file(file_path)

        qmp_log_df = pd.concat(qmp_log_data_list, axis=0)
        order_log_df = pd.concat(order_log_data_list, axis=0)
        stoppage_log_df = pd.concat(stoppage_log_data_list, axis=0)
        stoppage_event_df = pd.concat(stoppage_event_data_list, axis=0)

        return qmp_log_df, order_log_df, stoppage_log_df, stoppage_event_df, temp_index_df

    @staticmethod
    def read_temp_index_file(file_path):
        df = pd.read_csv(file_path, names=[DATETIME_COLUMN, "Time", "ShellIndex", "WrapIndex"], header=0)

        df["ShellIndex"] = pd.to_numeric(df["ShellIndex"].replace("-", np.nan))
        df["ShellIndex"].interpolate(method='linear', inplace=True)
        df["ShellIndex"].bfill(inplace=True)

        df["WrapIndex"] = pd.to_numeric(df["WrapIndex"].replace("-", np.nan))
        df["WrapIndex"].interpolate(method='linear', inplace=True)
        df["WrapIndex"].bfill(inplace=True)

        df[DATETIME_COLUMN] = df[DATETIME_COLUMN].fillna(method="ffill")
        df[DATETIME_COLUMN] = df[DATETIME_COLUMN].str.replace("202$", "2021")

        df["Time"] = df["Time"].str.replace("בין ", "")
        df["Time"] = df["Time"].str.replace("[-]+.*", ":00")
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN] + ' ' + df["Time"], dayfirst=True)
        df = StrausDataBuilder.filter_df_by_dt(df, DATETIME_COLUMN)
        df.sort_values(by=[DATETIME_COLUMN], ascending=True, inplace=True)

        df.drop(columns=["Time"], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def read_qmp_log_file(file_path):
        df = pd.read_csv(file_path)
        df[DATETIME_COLUMN] = pd.to_datetime(df['TimeStmp'])
        df = df.drop(columns=['TimeStmp', 'SpValue'], axis=1)
        return df

    @staticmethod
    def filter_qmp_df(df):
        df = StrausDataBuilder.filter_by_plant_model_id(df)
        df = StrausDataBuilder.filter_by_qmp(df)
        df = StrausDataBuilder.filter_df_by_dt(df, DATETIME_COLUMN)
        # df = StrausDataBuilder.agg_mean_qmp_by_qmp_order_step(df)
        return df

    @staticmethod
    def filter_by_qmp(df):
        df = df[df.QmpId.isin(QMP_FILTER_LIST)]
        return df

    @staticmethod
    def filter_df_by_dt(df, time_column, min_dt=MIN_DATETIME, max_dt=MAX_DATETIME):
        df = df[(df[time_column] >= min_dt) & (df[time_column] <= max_dt)]
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
        df['ActualStrTime'] = pd.to_datetime(df['ActualStrTime'])
        df['ActualEndTime'] = pd.to_datetime(df['ActualEndTime'])
        return df

    @staticmethod
    def filter_stoppage_df(df):
        df = StrausDataBuilder.filter_by_plant_model_id(df)
        df.drop(columns=['PlantModelId', 'BlamePlantModelId'], inplace=True)
        df = StrausDataBuilder.filter_df_by_dt(df, 'ActualStrTime')
        return df

    @staticmethod
    def read_order_log_file(file_path):
        df = pd.read_csv(file_path)
        df['ActualStrTime'] = pd.to_datetime(df['ActualStrTime'])
        df['ActualEndTime'] = pd.to_datetime(df['ActualEndTime'])
        return df

    @staticmethod
    def filter_order_log_df(df):
        df = StrausDataBuilder.filter_by_plant_model_id(df)
        df.drop(columns=['PlantModelId', 'OrderStepIdentity', 'ActualSetupTime', 'SetupTime', 'UomId'],
                inplace=True)
        df = StrausDataBuilder.filter_df_by_dt(df, "ActualStrTime")
        return df

    @staticmethod
    def read_stoppage_event_file(file_path):
        df = pd.read_csv(file_path)
        df = df.fillna(0)
        df = df.astype({"StoppageType": int})
        return df

    @staticmethod
    def filter_stoppage_event_df(df):
        df = df[df.StoppageType == UNPLANNED_STOPPAGE_TYPE_ID]
        df.drop(columns=['StoppageEventName', 'StoppageTypeName', "StoppageType"],
                inplace=True)
        return df

    @staticmethod
    def add_temp_index_to_qmp_order_log_data(qmp_order_log_df, temp_index_df):
        qmp_order_log_df["ShellIndex"] = np.nan
        qmp_order_log_df["WrapIndex"] = np.nan

        max_idx = temp_index_df.index.max()
        for idx, temp_index_row in temp_index_df.iterrows():
            start_time = temp_index_row[DATETIME_COLUMN]
            if idx + 1 <= max_idx:
                end_time = temp_index_df.loc[idx + 1, DATETIME_COLUMN]
            else:
                end_time = start_time + timedelta(hours=1)

            qmp_order_log_sub_df = StrausDataBuilder.filter_df_by_dt(qmp_order_log_df,
                                                                     DATETIME_COLUMN,
                                                                     start_time,
                                                                     end_time)

            qmp_order_log_df.loc[qmp_order_log_sub_df.index, "ShellIndex"] = temp_index_row["ShellIndex"]
            qmp_order_log_df.loc[qmp_order_log_sub_df.index, "WrapIndex"] = temp_index_row["WrapIndex"]

        qmp_order_log_df.dropna(inplace=True)
        return qmp_order_log_df

    def filter_night_mode(self, qmp_log_df):
        filter_df = qmp_log_df[qmp_log_df["QmpId"] == QMP_ID_NIGHT_MODE].sort_values(by=DATETIME_COLUMN)
        filter_df = filter_df[filter_df[self.config.get("ValueKeyword")] > NIGHT_MODE_THRESHOLD_VALUE]
        filter_df['TimeDiff'] = filter_df[DATETIME_COLUMN] - filter_df[DATETIME_COLUMN].shift(1)
        filter_df['TimeDiff'] = filter_df['TimeDiff'].fillna(timedelta(hours=2))
        dt_to_filter_list = pd.to_datetime(filter_df[filter_df['TimeDiff'] >= timedelta(hours=1)][DATETIME_COLUMN])
        index_to_filter = pd.Index([])
        for dt_to_filter in dt_to_filter_list:
            start_dt = dt_to_filter - timedelta(hours=2, minutes=0)
            end_dt = dt_to_filter + timedelta(hours=2, minutes=0)
            c_index_to_filter = qmp_log_df[
                (qmp_log_df[DATETIME_COLUMN] >= start_dt) & (qmp_log_df[DATETIME_COLUMN] <= end_dt)].index
            index_to_filter = index_to_filter.union(c_index_to_filter)
        qmp_log_df = qmp_log_df.drop(index=index_to_filter)
        return qmp_log_df

    @staticmethod
    def add_stoppage_ind_to_qmp_order_log_data(qmp_order_log_df, stoppage_log_event_df):
        qmp_order_log_df['is_stoppage'] = 0

        for idx, stoppage_log_event in stoppage_log_event_df.iterrows():
            stoppage_start_time = stoppage_log_event['ActualStrTime']
            stoppage_end_time = stoppage_log_event['ActualEndTime']
            stoppage_qmp_df = StrausDataBuilder.filter_df_by_dt(qmp_order_log_df,
                                                                DATETIME_COLUMN,
                                                                stoppage_start_time,
                                                                stoppage_end_time)

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
        qmp_order_log_joined_df.dropna(inplace=True)
        qmp_order_log_joined_df.drop(columns=["ActualStrTime", "ActualEndTime", "ActualQty"], inplace=True)
        qmp_order_log_joined_df['OrderStepId'] = qmp_order_log_joined_df.index
        qmp_order_log_joined_df = qmp_order_log_joined_df.astype({'OrderStepId': 'int32',
                                                                  "QmpId": "int32",
                                                                  "PartId": "int32"})
        qmp_order_log_joined_df.sort_values(by=DATETIME_COLUMN, inplace=True)
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

    def update_bounds(self, train_df, val_df, test_df):
        self.config['AnomalyConfig'][os.getenv("DATASET")] = {}
        groups = pd.unique(test_df[self.config.get("GroupKeyword")])
        for group in groups:
            self.config['AnomalyConfig'][os.getenv("DATASET")][group] = {}
            lower_bound = pd.unique(test_df[test_df[self.config.get("GroupKeyword")] == group]["MinValue"])[0]
            upper_bound = pd.unique(test_df[test_df[self.config.get("GroupKeyword")] == group]["MaxValue"])[0]
            group_bounds = [lower_bound, upper_bound]
            add_bounds_to_config(self.config, group, group_bounds, is_observed=True)

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

    def _assign_time_idx_per_group(self, df):
        df['time_idx'] = None
        keys = pd.unique(df[KEY_COLUMN])
        for key in keys:
            sub_df = df[df[KEY_COLUMN] == key]
            time_index = pd.unique(sub_df[DATETIME_COLUMN])
            time_index_time_idx_mapping = dict(zip(pd.to_datetime(time_index), list(range(1, len(time_index) + 1))))
            df.loc[sub_df.index, 'time_idx'] = sub_df.apply(
                lambda x: self._get_time_idx(x, time_index_time_idx_mapping), axis=1)
        return df

    @staticmethod
    def get_key_sub_df(data, key):
        plant_model_id, order_step_id, qmp_id = key.split(KEY_DELIMITER)
        key_sub_df = data[(data.PlantModelId == plant_model_id) &
                          (data.OrderStepId == order_step_id) &
                          (data.QmpId == qmp_id)]
        return key_sub_df

    def define_regression_ts_ds(self, df):
        ts_ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=["QmpId"],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            # static_categoricals=["QmpId"],
            static_reals=[],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            time_varying_known_reals=["time_idx", "TargetValue"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[self.config.get("ValueKeyword")],
            add_relative_time_idx=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={
                # self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True),
                "QmpId": NaNLabelEncoder(add_nan=True),
                **{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                   self.config.get("DatetimeAdditionalColumns")}},
            # target_normalizer=MultiNormalizer([NaNLabelEncoder(add_nan=True), NaNLabelEncoder(add_nan=True)])
            # target_normalizer=NaNLabelEncoder(add_nan=True)
        )
        return ts_ds

    def define_classification_ts_ds(self, df):
        ts_ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.config.get("ExceptionKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=[self.config.get("GroupKeyword")],
            static_reals=[],
            time_varying_known_categoricals=self.config.get("DatetimeAdditionalColumns"),
            time_varying_known_reals=["time_idx", "TargetValue"],
            time_varying_unknown_categoricals=[self.config.get("ExceptionKeyword")],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True),
                                  **{dt_col: NaNLabelEncoder(add_nan=True) for dt_col in
                                     self.config.get("DatetimeAdditionalColumns")}},
        )
        return ts_ds

    # def split_train_val_test(self, data: pd.DataFrame()):
    #     np.random.seed(42)
    #
    #     train_df_list = []
    #     val_df_list = []
    #     test_df_list = []
    #
    #     date_index = pd.DatetimeIndex(data[DATETIME_COLUMN])
    #     date_index_total_time = date_index.max() - date_index.min()
    #     test_time_start_dt = date_index.min() + date_index_total_time * (self.train_ratio + self.val_ratio)
    #
    #     test_df = data[lambda x: x[DATETIME_COLUMN] >= test_time_start_dt]
    #     data = data[lambda x: x[DATETIME_COLUMN] < test_time_start_dt]
    #
    #     encoder_len = self.config.get("EncoderLength")
    #     prediction_len = self.config.get("PredictionLength")
    #     groups = pd.unique(data[self.config.get("GroupKeyword")])
    #     for i, group in enumerate(groups):
    #         sub_df = data[data[self.config.get("GroupKeyword")] == group].reset_index(drop=True)
    #         if sub_df.index.max() <= encoder_len + prediction_len:
    #             continue
    #
    #         random_time_idx_list = list(np.random.choice(np.arange(sub_df.time_idx.min() + encoder_len,
    #                                                                sub_df.time_idx.max() - prediction_len),
    #                                                      size=(50, 2)))
    #         for time_idx in random_time_idx_list:
    #             train_sub_df = sub_df[(sub_df.time_idx >= time_idx[0] - encoder_len)
    #                                   & (sub_df.time_idx <= time_idx[0] + prediction_len)]
    #             val_sub_df = sub_df[(sub_df.time_idx >= time_idx[1] - encoder_len)
    #                                 & (sub_df.time_idx <= time_idx[1] + prediction_len)]
    #
    #             train_df_list.append(train_sub_df)
    #             val_df_list.append(val_sub_df)
    #
    #     # groups = pd.unique(test_df[self.config.get("GroupKeyword")])
    #     # for i, group in enumerate(groups):
    #     #     sub_df = data[data[self.config.get("GroupKeyword")] == group].reset_index(drop=True)
    #     #     if sub_df.index.max() <= encoder_len + prediction_len:
    #     #         continue
    #     #     else:
    #     #         test_df_list.append(sub_df)
    #
    #     train_df = pd.concat(train_df_list, axis=0)
    #     val_df = pd.concat(val_df_list, axis=0)
    #     # test_df = pd.concat(test_df_list, axis=0)
    #
    #     train_df = train_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']). \
    #         sort_values('time_idx'). \
    #         reset_index(drop=True)
    #     val_df = val_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']). \
    #         sort_values('time_idx'). \
    #         reset_index(drop=True)
    #     test_df = test_df.drop_duplicates(subset=['OrderStepId', 'QmpId', 'time_idx']). \
    #         sort_values('time_idx'). \
    #         reset_index(drop=True)
    #
    #     return train_df, val_df, test_df
