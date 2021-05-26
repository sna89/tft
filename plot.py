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
