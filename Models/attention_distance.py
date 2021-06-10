from config import DataConst
from typing import List
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


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
    return np.sqrt(1 - calc_rho_coef(p, q)) / float(DataConst.PREDICTION_LENGTH)


def init_zero_list(n):
    return [0] * n


def get_test_time_idx(test_dataloader):
    return test_dataloader.dataset.data['time'].unique()[DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH - 1:]


def get_sensor_test_values(test_df, sensor_id):
    return list(test_df[test_df.Sensor == sensor_id]['Value'])[DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH - 1:]


def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1)).T[0]
    return scaled_data


def calc_attention_dist(model, dataloader, df):
    raw_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
    sensor_id_indices_mapping = dataloader.dataset.decoded_index.groupby('Sensor').indices

    prediction_horizon_attention_mapping = create_prediction_horizon_attention_mapping(model, raw_predictions)

    for sensor_id, sensor_indices in sensor_id_indices_mapping.items():
        dist_t = list()
        T = len(sensor_indices)
        alpha_hat_prediction_horizon = dict()
        alpha_t_prediction_horizon = dict()

        for prediction_horizon, attention in prediction_horizon_attention_mapping.items():
            for t in range(T):
                alpha_hat_prediction_horizon[prediction_horizon] = list(map(
                    lambda curr_pos: attention[
                        sensor_indices, curr_pos + DataConst.ENCODER_LENGTH].mean()
                    if curr_pos <= prediction_horizon - 1
                    else torch.Tensor([0]),
                    range(-DataConst.ENCODER_LENGTH, DataConst.PREDICTION_LENGTH))
                )

                alpha_t_prediction_horizon[(t, prediction_horizon)] = list(map(
                    lambda curr_pos: attention[sensor_indices[t]][curr_pos + DataConst.ENCODER_LENGTH]
                    if curr_pos <= prediction_horizon - 1
                    else torch.Tensor([0]),
                    range(-DataConst.ENCODER_LENGTH, DataConst.PREDICTION_LENGTH))
                )

        for t in range(T):
            dist = torch.Tensor([0])
            for prediction_horizon in range(DataConst.PREDICTION_LENGTH):
                alpha_hat = alpha_hat_prediction_horizon[prediction_horizon]
                alpha_t = alpha_t_prediction_horizon[(t, prediction_horizon)]
                dist += calc_dist(alpha_hat,
                                  alpha_t)
            dist_t.append(dist.item())

        test_time_idx = get_test_time_idx(dataloader)
        sensor_test_values = get_sensor_test_values(df, sensor_id)
        scaled_sensor_test_values = scale_data(sensor_test_values)
        scaled_dist_t = scale_data(dist_t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_time_idx, y=scaled_sensor_test_values,
                                 mode='lines',
                                 name=sensor_id + '_scaled_test_data'))
        fig.add_trace(go.Scatter(x=test_time_idx, y=sensor_test_values,
                                 mode='lines',
                                 name=sensor_id + '_test_data'))
        fig.add_trace(go.Scatter(x=test_time_idx, y=dist_t,
                                 mode='lines',
                                 name=sensor_id + '_attention_dist'))
        fig.add_trace(go.Scatter(x=test_time_idx, y=scaled_dist_t,
                                 mode='lines',
                                 name=sensor_id + '_scaled_attention_dist'))
        fig.write_html(sensor_id + '_attention_dist.html')
