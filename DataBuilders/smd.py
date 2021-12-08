from datetime import timedelta
import os
import pandas as pd
from data_utils import add_dt_columns, assign_time_idx, create_bounds_labels
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from DataBuilders.data_builder import DataBuilder
from config import DATETIME_COLUMN, KEY_DELIMITER
import numpy as np

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
TEST_LABELS_FOLDER = "test_label"
NUM_DIMENSIONS = 38
COLUMNS = list(range(NUM_DIMENSIONS))
VAR_NAME = "Dimension"


class SMDDataBuilder(DataBuilder):
    def __init__(self, config):
        super(SMDDataBuilder, self).__init__(config)

    def get_machines_filenames(self):
        machines_names = os.listdir(os.path.join(self.config.get("Path"), TRAIN_FOLDER))
        return machines_names

    def _add_label_to_test_data(self, data, machine_filename):
        full_path = os.path.join(self.config.get("Path"), TEST_LABELS_FOLDER, machine_filename)
        label = pd.read_csv(full_path, header=None)
        data[self.config.get("LabelKeyword")] = label.values
        data[self.config.get("LabelKeyword")] = pd.to_numeric(data[self.config.get("LabelKeyword")], downcast='integer')
        return data

    def build_data(self):
        dfs = []
        machines_filenames = self.get_machines_filenames()
        for machine_filename in machines_filenames:
            machine_name, _ = machine_filename.split(".")
            for folder_name in [TRAIN_FOLDER] + [TEST_FOLDER]:
                full_path = os.path.join(self.config.get("Path"), folder_name, machine_filename)
                data = pd.read_csv(full_path, sep=",", names=COLUMNS)
                data["time_idx"] = data.index
                if folder_name == TEST_FOLDER:
                    data = self._add_label_to_test_data(data, machine_filename)
                    data = pd.melt(data,
                                   id_vars=["time_idx", self.config.get("LabelKeyword")],
                                   value_vars=COLUMNS,
                                   var_name=VAR_NAME,
                                   value_name=self.config.get("ValueKeyword"))
                else:
                    data = pd.melt(data,
                                   id_vars=["time_idx"],
                                   value_vars=COLUMNS,
                                   var_name=VAR_NAME,
                                   value_name=self.config.get("ValueKeyword"))
                data["type"] = folder_name
                data[self.config.get("GroupColumns")[0]] = machine_name
                dfs.append(data)
        data = pd.concat(dfs, axis=0)
        return data

    def preprocess(self, data):
        return data

    def define_regression_ts_ds(self, df):
        reg_ts_ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=self.config.get("GroupColumns"),
            min_encoder_length=self.config.get("EncoderLength"),
            max_encoder_length=self.config.get("EncoderLength"),
            min_prediction_length=self.config.get("PredictionLength"),
            max_prediction_length=self.config.get("PredictionLength"),
            static_categoricals=self.config.get("GroupColumns"),
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                self.config.get("ValueKeyword")
            ],
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
        )
        return reg_ts_ds

    def split_train_val_test(self, data: pd.DataFrame()):
        train_dfs = []
        val_dfs = []

        train_df = data[data["type"] == TRAIN_FOLDER]

        train_val_ratio = self.config.get("Train").get("TrainRatio") + self.config.get("Train").get("ValRatio")
        machines_names = list(pd.unique(data[self.config.get("GroupColumns")[0]]))
        for machine_name in machines_names:
            train_machine_df = train_df[train_df[self.config.get("GroupColumns")[0]] == machine_name]
            max_train_time_idx = int(train_machine_df["time_idx"].max() * train_val_ratio)
            train_dfs.append(train_machine_df[train_machine_df["time_idx"] <= max_train_time_idx])
            val_dfs.append(train_machine_df[train_machine_df["time_idx"] > max_train_time_idx])

        train_df = pd.concat(train_dfs, axis=0)
        train_df.reset_index(drop=True, inplace=True)

        val_df = pd.concat(val_dfs, axis=0)
        val_df.reset_index(drop=True, inplace=True)

        test_df = data[data["type"] == TEST_FOLDER]
        test_df.reset_index(drop=True, inplace=True)

        return train_df, val_df, test_df
