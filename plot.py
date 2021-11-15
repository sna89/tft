import matplotlib.pyplot as plt
import plotly.express as px
from pytorch_forecasting import Baseline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from config import DATETIME_COLUMN, PLOT
from data_utils import get_dataloader
from Models.trainer import get_prediction_mode

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


def plot_groups_attention(config, index_df, model_predictions, dataset_name):
    attention_dfs = []
    groups = set(map(lambda x: x.split("_")[0], list(index_df[config.get("GroupKeyword")])))
    for group in groups:
        group_indices = index_df[index_df[config.get("GroupKeyword")].str[:6] == group].index
        group_attention = model_predictions["attention"][group_indices, 0, :,
                           : model_predictions["encoder_lengths"].max()].mean(1).mean(0)
        attention_df = pd.DataFrame.from_dict({"group": group, "data": group_attention})
        attention_dfs.append(attention_df)
    attention_df = pd.concat(attention_dfs, axis=0)
    fig = px.line(attention_df, x=attention_df.index, y="data", color='group')
    fig.write_html(os.path.join(PLOT,
                                dataset_name,
                                "groups_attention.html"))


def plot_reg_predictions(config, model, df, ts_ds, dataset_name, model_name="TFT"):
    dataloader = get_dataloader(ts_ds, False, config)

    prediction_mode = get_prediction_mode()
    model_predictions, x = model.predict(dataloader, mode=prediction_mode, return_x=True, show_progress_bar=True)

    index_df = dataloader.dataset.x_to_index(x)

    if model_name == "TFT":
        plot_groups_attention(config, index_df, model_predictions, os.getenv("DATASET"))

    if model_name == "TFT":
        interpretation = model.interpret_output(model_predictions, reduction="sum", attention_prediction_horizon=0)
        figs = model.plot_interpretation(interpretation)
        figs['attention'].figure.savefig(os.path.join(PLOT, dataset_name, 'attention.png'))
        figs['static_variables'].figure.savefig(os.path.join(PLOT, dataset_name, 'static_variables.png'))
        figs['encoder_variables'].figure.savefig(os.path.join(PLOT, dataset_name, 'encoder_variables.png'))
        figs['decoder_variables'].figure.savefig(os.path.join(PLOT, dataset_name, 'decoder_variables.png'))

    if model_name == "TFT":
        predictions = model_predictions['prediction']
        interpretation = model.interpret_output(model_predictions, attention_prediction_horizon=0)

    elif model_name == "DeepAR":
        predictions = model_predictions

    idx_list = list(range(index_df.index.min(), index_df.index.max(), config.get("PredictionLength")))

    for idx in idx_list:
        plot_single_prediction(config, df, index_df, idx, predictions, x, model_name, dataset_name, interpretation)


def plot_single_prediction(config, df, index_df, idx, predictions, x, model_name, dataset_name, interpretation=None):
    time_idx, group = index_df.iloc[idx].values
    sub_df = df[df[config.get("GroupKeyword")] == group]
    x_values_enc = pd.DatetimeIndex(
        sub_df[(sub_df.time_idx <= time_idx) & (sub_df.time_idx >= time_idx - config.get("EncoderLength"))][
            DATETIME_COLUMN].unique())
    x_values_pred = pd.DatetimeIndex(
        sub_df[(sub_df.time_idx > time_idx) & (sub_df.time_idx <= time_idx + config.get("PredictionLength"))][
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

    prediction_idx = predictions[0].shape[1] // 2
    prediction = predictions[idx][:, prediction_idx]
    # prediction = predictions[idx]
    fig.add_trace(
        go.Scatter(x=x_values_pred, y=prediction, name="Prediction", line=dict(color='firebrick', width=4)),
        secondary_y=False,
    )

    if model_name == "TFT" and interpretation:
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

    fig.write_image(os.path.join(PLOT,
                                 dataset_name,
                                 'prediction_sensor_' + str(group) + '_' + str(time_idx) + '.png'),
                    engine='kaleido')


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
    if "time_idx" in data.columns:
        data = data.drop_duplicates(subset=["time_idx"] + [config.get("GroupKeyword")])

    if dataset_name == 'Synthetic':
        plot_synthetic_data(data)
    elif dataset_name == 'Fisherman':
        plot_fisherman_data(config, data)
    elif dataset_name == 'Fisherman2':
        plot_fisherman_data(config, data)
    elif dataset_name == 'Straus':
        plot_straus_data(dataset_name, data)


def plot_synthetic_data(data_to_plot):
    plot_name = "Synthetic_dataset.html"
    fig = px.line(data_to_plot, y="value", x=DATETIME_COLUMN, color='series')
    fig.write_html(plot_name)


def plot_fisherman_data(config, data_to_plot):
    fig = px.line(data_to_plot, y=config.get("ValueKeyword"), x=DATETIME_COLUMN, color=config.get("GroupKeyword"))
    fig.write_html(os.path.join(PLOT, 'Fisherman2', 'data.html'))


def plot_straus_data(dataset_name, data_to_plot):
    # for qmp_id in pd.unique(data_to_plot['QmpId']):
    #     sub_df = data_to_plot[data_to_plot.QmpId == qmp_id][['ActualValue', 'time_idx', 'key', DATETIME_COLUMN]]
    #     sub_df = sub_df.sort_values(by=['key', DATETIME_COLUMN], ascending=True)
    #     fig = px.line(sub_df, y="ActualValue", x=DATETIME_COLUMN, color='key')
    #     fig.write_html('straus_data_qmp_id_{}.html'.format(qmp_id))
    for order_step_id in pd.unique(data_to_plot['OrderStepId']):
        sub_df = data_to_plot[data_to_plot.OrderStepId == order_step_id][
            ['ActualValue', 'time_idx', 'key', DATETIME_COLUMN, 'is_stoppage']]
        sub_df = sub_df.sort_values(by=['key', DATETIME_COLUMN], ascending=True)
        fig = px.line(sub_df, y="ActualValue", x=DATETIME_COLUMN, color='key')
        stoppage_df = sub_df[[DATETIME_COLUMN, "is_stoppage"]].drop_duplicates().sort_values(by=DATETIME_COLUMN)
        fig.add_trace(go.Scatter(x=stoppage_df[DATETIME_COLUMN], y=stoppage_df["is_stoppage"],
                                 mode='lines',
                                 name='stoppage'))
        fig.write_html(os.path.join(PLOT, dataset_name, 'straus_data_order_step_id_{}.html'.format(order_step_id)))
    # fig = px.line(data_to_plot, y="ActualValue", x=DATETIME_COLUMN, color='key')
    # fig.write_html('straus_data.html')