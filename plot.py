import matplotlib.pyplot as plt
import plotly.express as px
from pytorch_forecasting import Baseline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from config import DATETIME_COLUMN


BASE_FOLDER = os.path.join("tmp", "pycharm_project_99")


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


def plot_predictions(model, dataloader, df, config, dataset_name, model_name="TFT"):
    if model_name == "TFT":
        model_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
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
                                                               config.get("PredictionLength")
                                                               )))].index)

    if os.getenv("DATASET") == "Fisherman" and model_name == "TFT":
        attention_dfs = []
        for sensor in list(index_df["Sensor"].unique()):
            sensor_indices = index_df[index_df.Sensor == sensor].index
            sensor_attention = model_predictions["attention"][sensor_indices, 0, :, : model_predictions["encoder_lengths"].max()].mean(1).mean(0)
            attention_df = pd.DataFrame.from_dict({"sensor": sensor, "data": sensor_attention})
            attention_dfs.append(attention_df)
        attention_df = pd.concat(attention_dfs, axis=0)
        fig = px.line(attention_df, x=attention_df.index, y="data", color='sensor')
        fig.write_html(os.path.join('Plots',
                                     dataset_name,
                                     "attention_sensors.html"))

    for idx in idx_list:
        time_idx, sensor = index_df.iloc[idx].values
        x_values_enc = pd.DatetimeIndex(
            df[(df.time_idx <= time_idx) & (df.time_idx >= time_idx - config.get("EncoderLength"))][DATETIME_COLUMN].unique())
        x_values_pred = pd.DatetimeIndex(
            df[(df.time_idx > time_idx) & (df.time_idx <= time_idx + config.get("PredictionLength"))][DATETIME_COLUMN].unique())

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

        fig.write_image(os.path.join('Plots',
                                     dataset_name,
                                     'prediction_sensor_' + str(sensor) + '_' + str(time_idx) + '.png'),
                        engine='kaleido')

    if model_name == "TFT":
        interpretation = model.interpret_output(model_predictions, reduction="sum", attention_prediction_horizon=0)
        figs = model.plot_interpretation(interpretation)
        figs['attention'].figure.savefig(os.path.join('Plots', dataset_name, 'attention.png'))
        figs['static_variables'].figure.savefig(os.path.join('Plots', dataset_name, 'static_variables.png'))
        figs['encoder_variables'].figure.savefig(os.path.join('Plots', dataset_name, 'encoder_variables.png'))
        figs['decoder_variables'].figure.savefig(os.path.join('Plots', dataset_name, 'decoder_variables.png'))


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


# def plot_synthetic_predictions(model, test_dataloader):
#     plot_name = '{Series}_series_{Seasonality}_seasonality_{Trend}_trend'.format(
#         Series=Params.SERIES,
#         Seasonality=Params.SEASONALITY,
#         Trend=Params.TREND
#     ).replace('.', '')
#
#     raw_predictions, x = model.predict(test_dataloader, mode="raw", return_x=True)
#     index_df = test_dataloader.dataset.x_to_index(x)
#     idx_list = list(index_df[index_df.time_idx.isin([460, 520, 580])].index)
#     for idx in idx_list:  # plot 10 examples
#         time_idx, series = index_df.loc[idx, ['time_idx', 'series']].values
#         model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
#         plt.savefig('plots/prediction_' + str(time_idx) + '_' + str(series))
#         plt.show()
#         plt.close()
#     interpretation = model.interpret_output(raw_predictions, reduction="sum", attention_prediction_horizon=0)
#     figs = model.plot_interpretation(interpretation)
#     figs['attention'].figure.savefig('plots/' + 'attention_' + plot_name)
#     figs['static_variables'].figure.savefig('plots/' + 'static_variables_' + plot_name)
#     figs['encoder_variables'].figure.savefig('plots/' + 'encoder_variables_' + plot_name)
#     figs['decoder_variables'].figure.savefig('plots/' + 'decoder_variables_' + plot_name)


def plot_data(config, dataset_name, data):
    data_to_plot = data.drop_duplicates(subset=["time_idx", config.get("GroupKeyword")])
    if dataset_name == 'Synthetic':
        plot_synthetic_data(config, data_to_plot)
    elif dataset_name == 'Fisherman':
        plot_fisherman_data(data_to_plot)


def plot_synthetic_data(config, data_to_plot):
    plot_name = '{Series}_series_{Seasonality}_seasonality_{Trend}_trend.html'.format(
        Series=config.get("Series"),
        Seasonality=config.get("Seasonality"),
        Trend=config.get("Trend")
    )
    fig = px.line(data_to_plot, y="value", x="time_idx", color='series')
    fig.write_html(plot_name)


def plot_fisherman_data(data_to_plot):
    # data_ = data[data.Type == 'internaltemp']
    fig = px.line(data_to_plot, y="Value", x="time_idx", color='Sensor')
    fig.write_html('fisherman.html')
