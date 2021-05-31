import plotly.express as px
import matplotlib.pyplot as plt
from constants import DataConst
import plotly.express as px


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


def plot_predictions(model, test_dataloader):
    raw_predictions, x = model.predict(test_dataloader, mode="raw", return_x=True)
    # index_df = test_dataloader.dataset.x_to_index(x)
    # idx_list = list(index_df[index_df.time_idx == 793].index)
    # for idx in idx_list:  # plot 10 examples
    #     # print(index_df.loc[idx, ['agency', 'sku']].values)
    #     model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
    #     plt.show()
    #     plt.close()
    # for i in range(DataConst.PREDICTION_LENGTH):
    interpretation = model.interpret_output(raw_predictions, reduction="sum", attention_prediction_horizon=0)
    figs = model.plot_interpretation(interpretation)
    figs['attention'].figure.savefig('plots/' + '5_series_10_seasonality_05_trend' + str(0))

    interpretation = model.interpret_output(raw_predictions, reduction="sum", attention_prediction_horizon=1)
    figs = model.plot_interpretation(interpretation)
    figs['attention'].figure.savefig('plots/' + '5_series_10_seasonality_05_trend' + str(1))


def plot_synthetic_data(data):
    fig = px.line(data, y="value", x="time_idx", color='series')
    fig.write_html('5_series_10_seasonality_05_trend.html')