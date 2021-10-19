from data_factory import get_data_builder


def get_processed_data(config, dataset_name):
    data_builder = get_data_builder(config, dataset_name)
    data = data_builder.build_data()
    return data


def split_df(config, dataset_name, data):
    data_builder = get_data_builder(config, dataset_name)
    train_df, val_df, test_df = data_builder.split_df(data)
    return train_df, val_df, test_df


def convert_df_to_ts_data(config, dataset_name, df, parameters=None, type_="reg"):
    data_builder = get_data_builder(config, dataset_name)
    ts_ds, parameters = data_builder.build_ts_data(df, parameters, type_)
    return ts_ds, parameters