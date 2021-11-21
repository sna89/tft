import os
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from DataBuilders.data_builder import DataBuilder
import numpy as np
import ast

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
MUTUAL_DATA = TRAIN_FOLDER + "_" + TEST_FOLDER
NUM_DIMENSIONS = 55
COLUMNS = list(range(NUM_DIMENSIONS))
LABELED_ANOMALIES_FILE = "labeled_anomalies.csv"


class MSLDataBuilder(DataBuilder):
    def __init__(self, config):
        super(MSLDataBuilder, self).__init__(config)

    @staticmethod
    def get_channel_list(labeled_anomalies):
        channel_list = list(labeled_anomalies[labeled_anomalies["spacecraft"] == os.getenv("DATASET")]["chan_id"])
        return channel_list

    def _add_label_to_test_data(self, data, labeled_anomalies, channel):
        channel_anomalies_array = self.get_channel_anomaly_array(labeled_anomalies, channel)
        data[self.config.get("LabelKeyword")] = channel_anomalies_array
        data[self.config.get("LabelKeyword")] = pd.to_numeric(data[self.config.get("LabelKeyword")],
                                                              downcast="integer")
        return data

    @staticmethod
    def get_channel_anomaly_array(labeled_anomalies, channel):
        channel_anomalies = labeled_anomalies[(labeled_anomalies["spacecraft"] == os.getenv("DATASET")) &
                                              (labeled_anomalies["chan_id"] == channel)]
        anomaly_sequences = ast.literal_eval(channel_anomalies["anomaly_sequences"].values[0])
        length = int(channel_anomalies["num_values"])
        channel_anomalies_array = np.zeros(length)
        for anomaly_sequence in anomaly_sequences:
            channel_anomalies_array[anomaly_sequence[0]: anomaly_sequence[1] + 1] = 1
        return channel_anomalies_array

    def build_data(self):
        labeled_anomalies = pd.read_csv(os.path.join(self.config.get("Path"), LABELED_ANOMALIES_FILE))
        channel_list = self.get_channel_list(labeled_anomalies)

        dfs = []
        for channel in channel_list:
            for folder_name in [TRAIN_FOLDER] + [TEST_FOLDER]:
                data = np.load(os.path.join(self.config.get("Path"), folder_name, channel + ".npy"))
                data = pd.DataFrame(data)
                data["time_idx"] = data.index
                if folder_name == TEST_FOLDER:
                    data = self._add_label_to_test_data(data, labeled_anomalies, channel)
                data.rename(columns={0: self.config.get("ValueKeyword")}, inplace=True)
                data["Type"] = folder_name
                data[self.config.get("GroupKeyword")] = channel
                dfs.append(data)
        data = pd.concat(dfs, axis=0)
        for column in COLUMNS[1:] + [self.config.get("LabelKeyword")]:
            data[column] = pd.to_numeric(data[column], downcast="integer")
            data[column] = data[column].astype(str).astype("category")
        data.columns = data.columns.astype(str)
        return data

    def preprocess(self, data):
        data = self._add_test_time_idx_column(data)
        data = self._update_time_idx_for_test_data(data)
        # data = self._move_encoder_samples_from_train_to_test(data)
        return data

    def _update_time_idx_for_test_data(self, data):
        channel_names = list(pd.unique(data[self.config.get("GroupKeyword")]))
        for channel in channel_names:
            max_time_idx = data[(data["Type"] == TRAIN_FOLDER) &
                                (data[self.config.get("GroupKeyword")] == channel)]["time_idx"].max()
            data.loc[(data["Type"] == TEST_FOLDER) &
                     (data[self.config.get("GroupKeyword")] == channel), "time_idx"] += max_time_idx
        return data

    @staticmethod
    def _add_test_time_idx_column(data):
        data.loc[data["Type"] == TEST_FOLDER, 'test_time_idx'] = \
             data[data["Type"] == TEST_FOLDER]["time_idx"]
        data['test_time_idx'] = pd.to_numeric(data['test_time_idx'], downcast="integer")
        return data

    def _move_encoder_samples_from_train_to_test(self, data):
        encoder_length = self.config.get("EncoderLength")
        channel_names = list(pd.unique(data[self.config.get("GroupKeyword")]))
        for channel in channel_names:
            channel_train_max_time_idx = data[(data["Type"] == TRAIN_FOLDER) &
                                              (data[self.config.get("GroupKeyword")] == channel)]["time_idx"].max()
            data.loc[(data["Type"] == TRAIN_FOLDER) &
                     (data[self.config.get("GroupKeyword")] == channel) &
                     (data['time_idx'] > channel_train_max_time_idx - encoder_length), "Type"] = MUTUAL_DATA
            data.loc[(data["Type"] == MUTUAL_DATA) &
                     (data[self.config.get("GroupKeyword")] == channel) &
                     (data['time_idx'] > channel_train_max_time_idx - encoder_length), self.config.get(
                "LabelKeyword")] = "0.0"
        return data

    def define_regression_ts_ds(self, df):
        reg_ts_ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=[self.config.get("GroupKeyword")],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[str(c) for c in COLUMNS[1:]],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={**{str(c): NaNLabelEncoder(add_nan=True) for c in COLUMNS[1:]}},
        )
        return reg_ts_ds

    def split_train_val_test(self, data: pd.DataFrame()):
        train_dfs = []
        val_dfs = []

        train_df = data[data["Type"] == TRAIN_FOLDER]
        test_df = data[data["Type"] == TEST_FOLDER]

        train_val_ratio = self.config.get("Train").get("TrainRatio") + self.config.get("Train").get("ValRatio")

        channel_names = list(pd.unique(data[self.config.get("GroupKeyword")]))
        for channel in channel_names:
            train_channel_df = train_df[train_df[self.config.get("GroupKeyword")] == channel]
            max_train_time_idx = int(train_channel_df["time_idx"].max() * train_val_ratio)
            train_dfs.append(train_channel_df[train_channel_df["time_idx"] <= max_train_time_idx])
            val_dfs.append(train_channel_df[train_channel_df["time_idx"] > max_train_time_idx])

        train_df = pd.concat(train_dfs, axis=0)
        train_df.drop(columns=[self.config.get("LabelKeyword")], inplace=True)
        train_df.reset_index(drop=True, inplace=True)

        val_df = pd.concat(val_dfs, axis=0)
        val_df.drop(columns=[self.config.get("LabelKeyword")], inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        test_df.reset_index(drop=True, inplace=True)
        return train_df, val_df, test_df
