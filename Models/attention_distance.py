from constants import DataConst
from typing import List
import numpy as np
import torch


def create_prediction_horizon_attention_mapping(model, raw_predictions):
    prediction_horizon_attention_mapping = dict()
    for prediction_horizon in range(DataConst.PREDICTION_LENGTH):
        interpretation = model.interpret_output(raw_predictions,
                                                attention_prediction_horizon=prediction_horizon)
        attention = interpretation['attention']
        prediction_horizon_attention_mapping[prediction_horizon] = attention
    return prediction_horizon_attention_mapping


def calc_rho_coef(p: List, q: List):
    return np.sum(np.sqrt(np.multiply(np.array(p), np.array(q))))


def calc_dist(p: List, q: List):
    return np.sqrt(1 - calc_rho_coef(p, q))


def init_zero_list(n):
    return [0] * n


def calc_attention_dist(model, test_dataloader):
    raw_predictions, x = model.predict(test_dataloader, mode="raw", return_x=True)
    sensor_id_indices_mapping = test_dataloader.dataset.decoded_index.groupby('Sensor').indices

    prediction_horizon_attention_mapping = create_prediction_horizon_attention_mapping(model, raw_predictions)

    for sensor_id, sensor_indices in sensor_id_indices_mapping.items():
        dist_t = list()
        T = len(sensor_indices)
        alpha_hat_prediction_horizon = dict()
        alpha_t_prediction_horizon = dict()

        for prediction_horizon, attention in prediction_horizon_attention_mapping.items():
            for t in range(T):
                current_horizon = attention[sensor_indices].shape[1] - DataConst.ENCODER_LENGTH

                alpha_hat_prediction_horizon[prediction_horizon] = list(map(
                    lambda curr_pos: attention[
                        sensor_indices, curr_pos + DataConst.ENCODER_LENGTH].mean() if curr_pos <= current_horizon - 1
                    else torch.Tensor([0])
                    , range(-DataConst.ENCODER_LENGTH, DataConst.PREDICTION_LENGTH))
                )

                alpha_t_prediction_horizon[(t, prediction_horizon)] = list(map(
                    lambda curr_pos: attention[sensor_indices[t]][curr_pos + DataConst.ENCODER_LENGTH]
                    if curr_pos <= current_horizon - 1
                    else torch.Tensor([0]),
                    range(-DataConst.ENCODER_LENGTH, DataConst.PREDICTION_LENGTH))
                )

        for t in range(T):
            dist = 0
            for prediction_horizon in range(DataConst.PREDICTION_LENGTH):
                alpha_hat = alpha_hat_prediction_horizon[prediction_horizon]
                alpha_t_prediction_horizon = alpha_t_prediction_horizon[(t, prediction_horizon)]
                dist += calc_dist(alpha_hat,
                                  alpha_t_prediction_horizon
                                  ) / float(DataConst.PREDICTION_LENGTH)
            dist_t.append(dist)
        print(sensor_id)
        print(dist_t)

