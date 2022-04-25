import pandas as pd
import numpy as np
from config import DATETIME_COLUMN
from DataBuilders.build import convert_df_to_ts_data
from Utils.data_utils import get_dataloader, get_group_id_group_name_mapping
from Utils.utils import flatten_nested_list
from plot import plot_single_prediction
import torch
from evaluation import get_classification_evaluation_summary
from DataBuilders.msl import LABELED_ANOMALIES_FILE
import os
from DataBuilders.msl import MSLDataBuilder


class AnomalyDetection:
    def __init__(self, config, model_name, dataset_name, forecast_model):
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.forecast_model = forecast_model

    def detect_and_evaluate(self, test_df, plot=False):
        num_anomalies_list = np.empty(0)
        test_ts_ds, _ = convert_df_to_ts_data(self.config, self.dataset_name, test_df, None, "reg")
        test_dl = get_dataloader(test_ts_ds, False, self.config)

        anomaly_dict = dict()
        anomaly_idx = 0

        predictions, x = self.forecast_model.predict(test_dl, return_x=True, mode='raw', show_progress_bar=True)
        x_index_df = test_dl.dataset.x_to_index(x)

        interpretation = self.forecast_model.interpret_output(predictions, attention_prediction_horizon=0)
        predictions = predictions['prediction']

        group_id_group_name_mapping = get_group_id_group_name_mapping(self.config, test_ts_ds)
        group_names = list(group_id_group_name_mapping.values())
        for i, group_id_tensor in enumerate(x['groups'].unique()):
            group_id = group_id_tensor.item()
            group_name = group_names[i]
            group_indices = torch.nonzero(x['groups'] == group_id).T[0]

            indices_to_ignore = []

            for idx in group_indices:
                if idx in indices_to_ignore:
                    continue

                prediction = predictions[idx]
                actual = x['decoder_target'][idx]

                num_anomalies, max_diff = self._detect(actual, prediction)
                num_anomalies_list = np.append(num_anomalies_list, num_anomalies)

                if max_diff > 0 and group_name != '53':
                    indices_to_ignore.extend(
                        self._get_prediction_indices(group_indices, idx.item(), self.config.get("PredictionLength") - 1)
                    )
                    current_time_idx = self.get_current_time_idx(test_df, group_name, idx, x)

                    current_dt = None
                    if DATETIME_COLUMN in test_df.columns:
                        current_dt = test_df[test_df['time_idx'] == current_time_idx][DATETIME_COLUMN].iloc[0]

                    anomaly_dict[anomaly_idx] = {
                        "GroupName": group_name,
                        "Actual": [actual_value.item() for actual_value in actual],
                        "LowQuantiles": [quantile[0].item() for quantile in prediction],
                        "HighQuantiles": [quantile[-1].item() for quantile in prediction],
                        "Time": current_time_idx if not current_dt else current_dt,
                        "NumExceptions": num_anomalies,
                        "MaxDiff": max_diff
                    }

                    anomaly_idx += 1
                    if plot:
                        plot_single_prediction(self.config,
                                               test_df,
                                               x_index_df,
                                               idx.item(),
                                               predictions,
                                               x,
                                               self.model_name,
                                               self.dataset_name,
                                               interpretation)
                else:
                    pass

        # filtered_num_anomalies_list = num_anomalies_list[num_anomalies_list > 0]
        anomaly_df = pd.DataFrame.from_dict(anomaly_dict, orient="index")
        anomaly_df.to_csv("anomalies.csv")
        return anomaly_df

    def get_current_time_idx(self, df, group_name, idx, x):
        current_time_idx = None
        if self.dataset_name == "MSL":
            current_time_idx = int(df[(df[self.config.get("GroupKeyword")] == group_name)
                   & (df["time_idx"] == x['decoder_time_idx'][idx][0].item())]
                ["test_time_idx"])
        elif self.dataset_name == "Synthetic":
            current_time_idx = x['decoder_time_idx'][idx][0].item()
        elif self.dataset_name == "Straus":
            time_idx_prediction_range = x['decoder_time_idx'][idx]
            time_idx_df_range = df[df[self.config.get("GroupKeyword")] == group_name]["time_idx"].values
            for time_idx in time_idx_prediction_range:
                time_idx = time_idx.item()
                if time_idx in time_idx_df_range:
                    current_time_idx = time_idx
                    break
        return current_time_idx

    def _detect(self, actual, prediction):
        anomalies = []
        max_diff = 0
        for idx, actual_value in enumerate(actual):
            lower_bound = prediction[idx][0]
            higher_bound = prediction[idx][-1]

            if actual_value < lower_bound or actual_value > higher_bound:
                c_max_diff = max(lower_bound - actual_value, actual_value - higher_bound)
                max_diff = max(c_max_diff.item(), max_diff)
                anomalies.append(idx)
                continue

        return len(anomalies), max_diff

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
