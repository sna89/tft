from data_factory import get_data_builder
from data_utils import get_dataloader


def get_processed_data(config, dataset_name):
    data_builder = get_data_builder(config, dataset_name)
    data = data_builder.build_data()
    return data


def split_df(config, dataset_name, data):
    data_builder = get_data_builder(config, dataset_name)
    train_df, val_df, test_df = data_builder.split_df(data, "time_idx", "ratio")
    return train_df, val_df, test_df


def convert_df_to_ts_data(config, dataset_name, df, parameters=None):
    data_builder = get_data_builder(config, dataset_name)
    ts_ds, parameters = data_builder.build_ts_data(df, parameters)
    return ts_ds, parameters


def build_exception_data(config, dataset_name, data):
    data_builder = get_data_builder(config, dataset_name)
    data_builder.split_df(data)

    train_exp_df, val_exp_df, test_exp_df, train_exp_ts_ds, val_exp_ts_ds, test_exp_ts_ds = data_builder.build_exception_ts_data()

    train_exp_dl = get_dataloader(train_exp_ts_ds, is_train=True, config=config)
    val_exp_dl = get_dataloader(val_exp_ts_ds, is_train=False, config=config)
    test_exp_dl = get_dataloader(test_exp_ts_ds, is_train=False, config=config)
    return train_exp_df, val_exp_df, test_exp_df, train_exp_ts_ds, val_exp_ts_ds, test_exp_ts_ds, train_exp_dl, val_exp_dl, test_exp_dl