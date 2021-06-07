import plotly.express as px
import matplotlib.pyplot as plt
from constants import DataConst, SyntheticDataSetParams
import plotly.express as px
from pytorch_forecasting import Baseline


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


def plot_fisherman_predictions(model, test_dataloader):
    plot_name = 'fisherman_predictions'
    raw_predictions, x = model.predict(test_dataloader, mode="raw", return_x=True)
    index_df = test_dataloader.dataset.x_to_index(x)
    idx_list = list(index_df[index_df.time_idx.isin(list(range(1591, 1769, 30)))].index)
    for idx in idx_list:
        time_idx, sensor = index_df.iloc[idx].values
        model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        plt.savefig('plots/prediction_sensor_{}_time_idx_{}'.format(sensor, time_idx))
        plt.show()
        plt.close()
    interpretation = model.interpret_output(raw_predictions, reduction="sum", attention_prediction_horizon=0)
    figs = model.plot_interpretation(interpretation)
    figs['attention'].figure.savefig('plots/' + 'attention_' + plot_name)
    figs['static_variables'].figure.savefig('plots/' + 'static_variables_' + plot_name)
    figs['encoder_variables'].figure.savefig('plots/' + 'encoder_variables_' + plot_name)
    figs['decoder_variables'].figure.savefig('plots/' + 'decoder_variables_' + plot_name)


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


def plot_synthetic_predictions(model, test_dataloader):
    plot_name = '{Series}_series_{Seasonality}_seasonality_{Trend}_trend'.format(
            Series=SyntheticDataSetParams.SERIES,
            Seasonality=SyntheticDataSetParams.SEASONALITY,
            Trend=SyntheticDataSetParams.TREND
        ).replace('.', '')

    raw_predictions, x = model.predict(test_dataloader, mode="raw", return_x=True)
    index_df = test_dataloader.dataset.x_to_index(x)
    idx_list = list(index_df[index_df.time_idx.isin([460, 520, 580])].index)
    for idx in idx_list:  # plot 10 examples
        time_idx, series = index_df.loc[idx, ['time_idx', 'series']].values
        model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        plt.savefig('plots/prediction_' + str(time_idx) + '_' + str(series))
        plt.show()
        plt.close()
    interpretation = model.interpret_output(raw_predictions, reduction="sum", attention_prediction_horizon=0)
    figs = model.plot_interpretation(interpretation)
    figs['attention'].figure.savefig('plots/' + 'attention_' + plot_name)
    figs['static_variables'].figure.savefig('plots/' + 'static_variables_' + plot_name)
    figs['encoder_variables'].figure.savefig('plots/' + 'encoder_variables_' + plot_name)
    figs['decoder_variables'].figure.savefig('plots/' + 'decoder_variables_' + plot_name)


def plot_data(dataset_name, data):
    if dataset_name == 'synthetic':
        plot_synthetic_data(data)
    elif dataset_name == '2_fisherman':
        plot_fisherman_data(data)


def plot_synthetic_data(data):
    plot_name = '{Series}_series_{Seasonality}_seasonality_{Trend}_trend'.format(
            Series=SyntheticDataSetParams.SERIES,
            Seasonality=SyntheticDataSetParams.SEASONALITY,
            Trend=SyntheticDataSetParams.TREND
        ).replace('.', '')

    fig = px.line(data, y="value", x="time_idx", color='series')
    fig.write_html(plot_name)


def plot_fisherman_data(data):
    # data_ = data[data.Type == 'internaltemp']
    fig = px.line(data, y="Value", x="time_idx", color='Sensor')
    fig.write_html('fisherman.html')