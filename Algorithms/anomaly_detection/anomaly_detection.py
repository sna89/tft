import pandas as pd
import numpy as np
from config import DATETIME_COLUMN, PLOT
from DataBuilders.build import convert_df_to_ts_data
from data_utils import get_dataloader
from plot import plot_single_prediction
import torch


class AnomalyDetection:
    def __init__(self, config, model_name, dataset_name, forecast_model):
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.forecast_model = forecast_model

    def detect(self, df, plot=False):
        anomaly_dict = dict()
        anomaly_idx = 0

        ts_ds, _ = convert_df_to_ts_data(self.config, self.dataset_name, df, parameters=None, type_="reg")

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
            indices_to_ignore = []

            for idx in group_indices:
                if idx in indices_to_ignore:
                    continue

                prediction = predictions[idx]
                actual = x['decoder_target'][idx]

                if self._is_out_of_bounds(actual, prediction):
                    indices_to_ignore.extend(self._get_prediction_indices(group_indices, idx.item(), self.config.get("PredictionLength")))
                    group_name = df.iloc[group_indices.min().item()][self.config.get("GroupKeyword")]
                    current_time_idx = x['decoder_time_idx'][idx][0].item()
                    current_dt = df[df['time_idx'] == current_time_idx][DATETIME_COLUMN].iloc[0]

                    anomaly_dict[anomaly_idx] = {
                        "GroupName": group_name,
                        "Actual": [actual_value.item() for actual_value in actual],
                        "LowQuantiles": [quantile[0].item() for quantile in prediction],
                        "HighQuantiles": [quantile[-1].item() for quantile in prediction],
                        "Time": current_dt,
                        # "SeriesStartTime": group_start_dt,
                        # "SeriesEndTime": group_end_dt,
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

        anomaly_df = pd.DataFrame.from_dict(anomaly_dict, orient="index")
        anomaly_df.to_csv("anomaly_df.csv")
        return anomaly_df

    @staticmethod
    def _is_out_of_bounds(actual, prediction):
        first_outer_quantile_exceptions = []
        second_outer_quantile_excpeptions = []
        for idx, actual_value in enumerate(actual):
            first_lower_bound = prediction[idx][0]
            first_higher_bound = prediction[idx][-1]

            second_lower_bound = prediction[idx][1]
            second_higher_bound = prediction[idx][-2]
            if actual_value < first_lower_bound or actual_value > first_higher_bound:
                first_outer_quantile_exceptions.append(idx)
                continue

            if actual_value < second_lower_bound or actual_value > second_higher_bound:
                second_outer_quantile_excpeptions.append(idx)

        if len(first_outer_quantile_exceptions) >= 3 or \
            (len(first_outer_quantile_exceptions) >= 2 and len(second_outer_quantile_excpeptions) >= 1) or \
                (len(first_outer_quantile_exceptions) >= 1 and len(second_outer_quantile_excpeptions) >= 2):
            return True
        return False

    @staticmethod
    def _get_prediction_indices(torch_indices, idx, horizon):
        prediction_max_idx = torch_indices.max().item() if idx + horizon >= torch_indices.max().item() else idx + horizon
        prediction_indices = list(range(idx, prediction_max_idx + 1))
        return prediction_indices






