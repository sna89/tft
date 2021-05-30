import numpy as np
from constants import DataConst


def preprocess_synthetic(data):
    add_month_column(data)
    add_day_column(data)


def preprocess_stallion(data):
    data.drop(['timeseries'], axis=1, inplace=True)
    add_time_idx_column(data)
    add_month_column(data)
    add_log_column(data, 'industry_volume')
    add_log_column(data, 'soda_volume')
    convert_special_days_to_categorical(data)


def add_time_idx_column(data):
    data['time_idx'] = data.date.dt.year * 12 + data.date.dt.month
    data['time_idx'] -= data.time_idx.min()


def add_month_column(data):
    data['month'] = data.date.dt.month.astype(str).astype("category")


def add_day_column(data):
    data['day'] = data.date.dt.day.astype(str).astype("category")


def add_log_column(data, col_name):
    new_col_name = 'log_' + col_name
    data[new_col_name] = np.log(data[col_name])


def convert_special_days_to_categorical(data):
    data[DataConst.SPECIAL_DAYS] = data[DataConst.SPECIAL_DAYS].apply(lambda x: x.map({0: "-", 1: x.name})).astype(
        "category")
