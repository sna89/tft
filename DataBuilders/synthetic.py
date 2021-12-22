import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, MultiNormalizer, NaNLabelEncoder
from data_utils import create_bounds_labels
from DataBuilders.data_builder import DataBuilder
import numpy as np
import os
from config import OBSERVED_KEYWORD, OBSERVED_LB_KEYWORD, OBSERVED_UB_KEYWORD, \
    NOT_OBSERVED_KEYWORD, NOT_OBSERVED_LB_KEYWORD, NOT_OBSERVED_UB_KEYWORD, update_config


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
        self._update_bounds(data)
        update_config(self.config)
        data = create_bounds_labels(self.config, data)
        return data

    def _update_bounds(self, data):
        self.config['AnomalyConfig'][os.getenv("DATASET")] = {}
        groups = pd.unique(data[self.config.get("GroupKeyword")])
        train_val_ratio = self.config.get("Train").get("TrainRatio") + self.config.get("Train").get("ValRatio")
        max_val_time_idx = int(data['time_idx'].max() * train_val_ratio)
        for group in groups:
            group_quantiles = np.quantile(
                data[(data[self.config.get("GroupKeyword")] == group) &
                     (data["time_idx"] < max_val_time_idx)][self.config.get("ValueKeyword")],
                q=[0, 0.025, 0.975, 1])

            self.config['AnomalyConfig'][os.getenv("DATASET")][group] = {}
            self._add_bounds_to_config(group, group_quantiles, is_observed=True)
            self._add_bounds_to_config(group, group_quantiles, is_observed=False)

    def _add_bounds_to_config(self, group, group_quantiles, is_observed=True):
        if is_observed:
            self._add_observed_bounds_to_config(group, group_quantiles)
        else:
            self._add_not_observed_bounds_to_config(group, group_quantiles)

    def _add_observed_bounds_to_config(self, group, group_quantiles):
        observed_lb, observed_ub = group_quantiles[1], group_quantiles[2]
        self.config['AnomalyConfig'][os.getenv("DATASET")][group][OBSERVED_KEYWORD] = {OBSERVED_LB_KEYWORD: observed_lb,
                                                                                       OBSERVED_UB_KEYWORD: observed_ub}

    def _add_not_observed_bounds_to_config(self, group, group_quantiles):
        delta = self._calc_bound_delta(group_quantiles)
        not_observed_lb, not_observed_ub = group_quantiles[0] - delta, group_quantiles[-1] + delta

        self.config['AnomalyConfig'][os.getenv("DATASET")][group][NOT_OBSERVED_KEYWORD] = {
            NOT_OBSERVED_LB_KEYWORD: not_observed_lb,
            NOT_OBSERVED_UB_KEYWORD: not_observed_ub}

    @staticmethod
    def _calc_bound_delta(group_quantiles):
        return abs(group_quantiles[0] - group_quantiles[-1]) * 0.1

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
