import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from Utils.data_utils import add_bounds_to_config
from DataBuilders.data_builder import DataBuilder
import numpy as np
import os


class SyntheticDataBuilder(DataBuilder):
    def __init__(self, config):
        super().__init__(config)

    def build_data(self):
        path = self.config.get("Path")
        filename = self.config.get("Filename")
        full_path = os.path.join(path, filename)
        data = pd.read_csv(full_path)
        data = data.drop(columns=['Unnamed: 0'], axis=1)
        return data

    def preprocess(self, data):
        data[self.config.get("GroupKeyword")] = data[self.config.get("GroupKeyword")].astype(str).astype("category")
        return data

    def update_bounds(self, train_df, val_df, test_df):
        self.config['AnomalyConfig'][os.getenv("DATASET")] = {}
        data = pd.concat([train_df, val_df], axis=0)
        groups = pd.unique(data[self.config.get("GroupKeyword")])
        for group in groups:
            group_quantiles = np.quantile(
                data[data[self.config.get("GroupKeyword")] == group][self.config.get("ValueKeyword")],
                q=[0, 0.025, 0.975, 1])

            self.config['AnomalyConfig'][os.getenv("DATASET")][group] = {}
            add_bounds_to_config(self.config, group, group_quantiles[1: 3], is_observed=True)
            add_bounds_to_config(self.config, group, group_quantiles[[0, -1]], is_observed=False)

    def define_regression_ts_ds(self, train_df):
        ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=self.config.get("ValueKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=[self.config.get("ValueKeyword")],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=[],
            static_reals=[],
            static_categoricals=[self.config.get("GroupKeyword")],
            # target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=False,
            randomize_length=None
        )
        return ts_ds

    def define_classification_ts_ds(self, train_df):
        ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=self.config.get("ObservedBoundKeyword"),
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=[self.config.get("ValueKeyword")],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=[],
            static_reals=[],
            static_categoricals=[self.config.get("GroupKeyword")],
            time_varying_unknown_categoricals=[self.config.get("ObservedBoundKeyword")],
            add_relative_time_idx=True,
            add_target_scales=False,
            randomize_length=None,
            # categorical_encoders={self.config.get("GroupKeyword"): NaNLabelEncoder(add_nan=True)}
        )
        return ts_ds

    def define_combined_ts_ds(self, train_df):
        ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=[self.config.get("ValueKeyword"), self.config.get("ObservedBoundKeyword")],
            group_ids=[self.config.get("GroupKeyword")],
            min_encoder_length=self.enc_length,
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=[self.config.get("ValueKeyword")],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=[],
            time_varying_unknown_categoricals=[self.config.get("ObservedBoundKeyword")],
            static_reals=[],
            static_categoricals=[self.config.get("GroupKeyword")],
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
        )
        return ts_ds
