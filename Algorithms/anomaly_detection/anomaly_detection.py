import pandas as pd
import numpy as np
from config import DATETIME_COLUMN, PLOT
from DataBuilders.build import convert_df_to_ts_data
from data_utils import get_dataloader, flatten_nested_list
from plot import plot_single_prediction
import torch
from evaluation import get_classification_evaluation_summary
from DataBuilders.msl import LABELED_ANOMALIES_FILE
import os
from DataBuilders.msl import MSLDataBuilder
import itertools


class AnomalyDetection:
    def __init__(self, config, model_name, dataset_name, forecast_model):
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.forecast_model = forecast_model

    def detect_and_evaluate(self, df, ts_ds, plot=False):
        anomaly_actual_dict = self._build_anomaly_actual_dict()
        anomaly_prediction_dict = dict()
        anomaly_dict = dict()
        anomaly_idx = 0

        dl = get_dataloader(ts_ds, False, self.config)

        predictions, x = self.forecast_model.predict(dl, return_x=True, mode='raw', show_progress_bar=True)
        x_index_df = dl.dataset.x_to_index(x)

        # if self.model_name == "TFT":
        interpretation = self.forecast_model.interpret_output(predictions, attention_prediction_horizon=0)
        predictions = predictions['prediction']

        groups = x['groups'].unique()
        for group in groups:
            mask = x['groups'] == group
            group_indices = torch.nonzero(mask)[:, 0]
            group_name = self._get_group_name(x_index_df, group_indices[0])

            anomaly_prediction_dict[group_name] = [0] * self.config.get("EncoderLength")
            indices_to_ignore = []

            for idx in group_indices:
                if idx in indices_to_ignore:
                    anomaly_prediction_dict[group_name].append(1)
                    continue

                prediction = predictions[idx]
                actual = x['decoder_target'][idx]

                if self._detect(actual, prediction):
                    indices_to_ignore.extend(
                        self._get_prediction_indices(group_indices, idx.item(), self.config.get("PredictionLength") - 1)
                    )
                    anomaly_prediction_dict[group_name].append(1)
                    current_time_idx = self.get_current_time_idx(df, group_name, idx, x)

                    current_dt = None
                    if DATETIME_COLUMN in df.columns:
                        current_dt = df[df['time_idx'] == current_time_idx][DATETIME_COLUMN].iloc[0]

                    anomaly_dict[anomaly_idx] = {
                        "GroupName": group_name,
                        "Actual": [actual_value.item() for actual_value in actual],
                        "LowQuantiles": [quantile[0].item() for quantile in prediction],
                        "HighQuantiles": [quantile[-1].item() for quantile in prediction],
                        "Time": current_time_idx if not current_dt else current_dt,
                    }
                    anomaly_idx += 1
                    if plot:
                        plot_single_prediction(self.config,
                                               df,
                                               x_index_df,
                                               idx.item(),
                                               predictions,
                                               x,
                                               self.model_name,
                                               self.dataset_name,
                                               interpretation)
                else:
                    anomaly_prediction_dict[group_name].append(0)

            last_prediction_anomaly = anomaly_prediction_dict[group_name][-1]
            anomaly_prediction_dict[group_name].extend(
                [last_prediction_anomaly] * (self.config.get("PredictionLength") - 1))

        anomaly_df = pd.DataFrame.from_dict(anomaly_dict, orient="index")
        self._evaluate(anomaly_actual_dict, anomaly_prediction_dict)
        return anomaly_df

    def get_current_time_idx(self, df, group_name, idx, x):
        current_time_idx = int(df[(df[self.config.get("GroupKeyword")] == group_name)
                                  & (df["time_idx"] == x['decoder_time_idx'][idx][0].item())]
                               ["test_time_idx"]) if \
            self.dataset_name == "MSL" else \
            x['decoder_time_idx'][idx][0].item()
        return current_time_idx

    def _detect(self, actual, prediction):
        first_outer_quantile_exceptions = []
        # second_outer_quantile_excpeptions = []
        for idx, actual_value in enumerate(actual):
            first_lower_bound = prediction[idx][0]
            first_higher_bound = prediction[idx][-1]

            # second_lower_bound = prediction[idx][1]
            # second_higher_bound = prediction[idx][-2]
            if actual_value < first_lower_bound or actual_value > first_higher_bound:
                first_outer_quantile_exceptions.append(idx)
                continue

            # if actual_value < second_lower_bound or actual_value > second_higher_bound:
            #     second_outer_quantile_excpeptions.append(idx)

        if len(first_outer_quantile_exceptions) > self.config.get("PredictionLength") // 2:
            #     or \
            # (len(first_outer_quantile_exceptions) >= 2 and len(second_outer_quantile_excpeptions) >= 1) or \
            #     (len(first_outer_quantile_exceptions) >= 1 and len(second_outer_quantile_excpeptions) >= 2):
            return True
        return False

    def _evaluate(self, anomaly_actual_dict, anomaly_prediction_dict):
        channels = list(anomaly_prediction_dict.keys())
        for channel in channels:
            print(channel)
            anomaly_channel_actual = np.array(anomaly_actual_dict[channel])
            anomaly_channel_prediction = np.array(anomaly_prediction_dict[channel])
            get_classification_evaluation_summary(anomaly_channel_actual, anomaly_channel_prediction)
        print("Total")
        anomaly_actual = np.array(flatten_nested_list([anomaly_channel_actual
                                                       for channel, anomaly_channel_actual
                                                       in anomaly_actual_dict.items()
                                                       if channel in channels]))
        anomaly_prediction = np.array(flatten_nested_list([anomaly_channel_prediction
                                                           for channel, anomaly_channel_prediction
                                                           in anomaly_prediction_dict.items()]))
        get_classification_evaluation_summary(anomaly_actual, anomaly_prediction)

    def _build_anomaly_actual_dict(self):
        if self.dataset_name == "MSL":
            msl_labeled_anomalies_dict = dict()
            labeled_anomalies = pd.read_csv(os.path.join(self.config.get("Path"), LABELED_ANOMALIES_FILE))
            channels = pd.unique(labeled_anomalies[labeled_anomalies["spacecraft"] == self.dataset_name]["chan_id"])
            for channel in channels:
                channel_anomaly_array = MSLDataBuilder.get_channel_anomaly_array(labeled_anomalies, channel)
                msl_labeled_anomalies_dict[channel] = channel_anomaly_array
            return msl_labeled_anomalies_dict

    @staticmethod
    def _get_prediction_indices(torch_indices, idx, horizon):
        prediction_max_idx = torch_indices.max().item() if idx + horizon >= torch_indices.max().item() else idx + horizon
        prediction_indices = list(range(idx, prediction_max_idx + 1))
        return prediction_indices

    def _get_group_name(self, x_index_df, idx):
        group_name = x_index_df.iloc[idx.item()][self.config.get("GroupKeyword")]
        return group_name