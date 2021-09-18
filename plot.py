import matplotlib.pyplot as plt
import plotly.express as px
from pytorch_forecasting import Baseline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from config import DATETIME_COLUMN
from utils import PLOT


def plot_volume_by_group(data, agency=None, sku=None):
    df = data.copy()

    df['agency_sku'] = df.agency.astype(str) + df.sku.astype(str)
    if agency:
        df = df[df.agency == agency]
    if sku:
        df = df[df.sku == sku]

    fig = px.line(df, x='date', y='volume', color='agency_sku')
    fig.show()
    del df


def plot_predictions(config, model, dataloader, df):
    model_name = config.get("model")

    if model_name == "TFT":
        model_predictions, x = model.predict(dataloader, mode="raw", return_x=True, show_progress_bar=True)
        predictions = model_predictions['prediction']
        interpretation = model.interpret_output(model_predictions, attention_prediction_horizon=0)

    elif model_name == "DeepAR":
        predictions, x = model.predict(dataloader, mode="quantiles", return_x=True)

    prediction_idx = predictions[0].shape[1] // 2
    index_df = dataloader.dataset.x_to_index(x)
    time_idx_min = index_df.time_idx.min()
    time_idx_max = index_df.time_idx.max()
    idx_list = list(index_df[index_df.time_idx.isin(list(range(time_idx_min,
                                                               time_idx_max,
                                                               config.get("Data").get("PredictionLength")
                                                               )))].index)

    if model_name == "TFT":
        attention_dfs = []
        for group in list(index_df[config.get("Data").get("GroupKeyword")].unique()):
            group_indices = index_df[index_df[config.get("Data").get("GroupKeyword")] == group].index
            group_attention = model_predictions["attention"][group_indices, 0, :,
                               : model_predictions["encoder_lengths"].max()].mean(1).mean(0)
            attention_df = pd.DataFrame.from_dict({"group": group, "data": group_attention})
            attention_dfs.append(attention_df)
        attention_df = pd.concat(attention_dfs, axis=0)
        fig = px.line(attention_df, x=attention_df.index, y="data", color='group')
        fig.write_html(os.path.join(PLOT,
                                    "attention_sensors.html"))

    for idx in idx_list:
        time_idx, group = index_df.iloc[idx].values
        sub_df = df[df[config.get("Data").get("GroupKeyword")] == group]
        x_values_enc = pd.DatetimeIndex(
            sub_df[(sub_df.time_idx <= time_idx) & (sub_df.time_idx >= time_idx - config.get("Data").get("EncoderLength"))][
                DATETIME_COLUMN].unique())
        x_values_pred = pd.DatetimeIndex(
            sub_df[(sub_df.time_idx > time_idx) & (sub_df.time_idx <= time_idx + config.get("Data").get("PredictionLength"))][
                DATETIME_COLUMN].unique())

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        actual_enc = x['encoder_target'][idx]
        fig.add_trace(
            go.Scatter(x=x_values_enc, y=actual_enc, name="Actual", line=dict(color='royalblue', width=1)),
            secondary_y=False,
        )

        actual_dec = x['decoder_target'][idx]
        fig.add_trace(
            go.Scatter(x=x_values_pred, y=actual_dec, name="Actual", line=dict(color='royalblue', width=4)),
            secondary_y=False,
        )

        prediction = predictions[idx][:, prediction_idx]
        fig.add_trace(
            go.Scatter(x=x_values_pred, y=prediction, name="Prediction", line=dict(color='firebrick', width=4)),
            secondary_y=False,
        )

        if model_name == "TFT":
            attention = interpretation["attention"][idx]
            fig.add_trace(
                go.Scatter(x=x_values_enc, y=attention, name="Attention", line=dict(color='lightgrey')),
                secondary_y=True,
            )

        y_lower_quantile = predictions[idx][:, 0]
        fig.add_trace(
            go.Scatter(x=x_values_pred, y=y_lower_quantile, name="Lower Quantile",
                       line=dict(color='grey', dash='dash')),
            secondary_y=False,
        )

        y_upper_quantile = predictions[idx][:, -1]
        fig.add_trace(
            go.Scatter(x=x_values_pred, y=y_upper_quantile, name="Upper Quantile", line=dict(color='grey', dash='dash'),
                       fill='tonexty'),
            secondary_y=False,
        )

        fig.update_xaxes(title_text="<b>Time</b>")
        fig.update_yaxes(title_text="<b>Actual VS Prediction</b>", secondary_y=False)
        if model_name == "TFT":
            fig.update_yaxes(title_text="<b>Attention</b>", secondary_y=True)
        fig.update_layout(height=800, width=1400)

        dt = sub_df[sub_df["time_idx"] == time_idx][DATETIME_COLUMN].values[0]
        fig.write_image(os.path.join('Plots',
                                     'prediction_sensor_' + str(group) + '_' + str(dt) + '.png'),
                        engine='kaleido')

    if model_name == "TFT":
        interpretation = model.interpret_output(model_predictions, reduction="sum", attention_prediction_horizon=0)
        figs = model.plot_interpretation(interpretation)
        figs['attention'].figure.savefig(os.path.join('Plots', 'attention.png'))
        figs['static_variables'].figure.savefig(os.path.join('Plots', 'static_variables.png'))
        figs['encoder_variables'].figure.savefig(os.path.join('Plots', 'encoder_variables.png'))
        figs['decoder_variables'].figure.savefig(os.path.join('Plots', 'decoder_variables.png'))


def plot_baseline_predictions(test_dataloader):
    baseline_predictions, x = Baseline().predict(test_dataloader, mode='raw', return_x=True)
    index_df = test_dataloader.dataset.x_to_index(x)
    idx_list = list(index_df[index_df.time_idx.isin(list(range(1591, 1769, 30)))].index)
    for idx in idx_list:
        time_idx, sensor = index_df.iloc[idx].values
        Baseline().plot_prediction(x, baseline_predictions, idx=idx, add_loss_to_title=True)
        plt.savefig('plots/prediction_sensor_{}_time_idx_{}'.format(sensor, time_idx))
        plt.show()
        plt.close()


def plot_data(config, data):
    data_to_plot = data.drop_duplicates(subset=["time_idx"] + [config.get("Data").get("GroupKeyword")])
    plot_fisherman_data(data_to_plot)


def plot_fisherman_data(data_to_plot):
    fig = px.line(data_to_plot, y="Value", x=DATETIME_COLUMN, color='Sensor')
    fig.write_html(os.path.join(PLOT, 'fisherman.html'))
