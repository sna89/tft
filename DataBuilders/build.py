from config import update_config
from data_factory import get_data_builder
from Utils.data_utils import create_bounds_labels


def build_data(config, dataset_name):
    data_builder = get_data_builder(config, dataset_name)
    data = data_builder.build_data()
    return data


def process_data(config, dataset_name, data):
    data_builder = get_data_builder(config, dataset_name)
    return data_builder.preprocess(data)


def update_bounds(config, dataset_name, train_df, val_df, test_df):
    data_builder = get_data_builder(config, dataset_name)
    data_builder.update_bounds(train_df, val_df, test_df)
    update_config(config)
    train_df = create_bounds_labels(config, train_df)
    val_df = create_bounds_labels(config, val_df)
    test_df = create_bounds_labels(config, test_df)
    return train_df, val_df, test_df


def split_df(config, dataset_name, data):
    data_builder = get_data_builder(config, dataset_name)
    train_df, val_df, test_df = data_builder.split_train_val_test(data)
    return train_df, val_df, test_df


def convert_df_to_ts_data(config, dataset_name, df, parameters, task_type):
    data_builder = get_data_builder(config, dataset_name)
    ts_ds, parameters = data_builder.build_ts_data(df, parameters, task_type)
    return ts_ds, parameters