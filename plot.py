import plotly.express as px
import matplotlib.pyplot as plt


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


def plot_predictions(model, val_dataloader, train_dataloader):
    raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True)
    # index_df = val_dataloader.dataset.x_to_index(x)
    # for idx in range(5):  # plot 10 examples
    #     print(index_df.loc[idx, ['agency', 'sku']].values)
    #     model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
    #     plt.show()
    #     plt.close()
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    model.plot_interpretation(interpretation)
    plt.show()
    plt.close()

    raw_predictions, x = model.predict(train_dataloader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    model.plot_interpretation(interpretation)
    plt.show()
    plt.close()