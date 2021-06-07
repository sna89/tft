import numpy as np
from constants import DataConst
import pandas as pd


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


def filter_df_by_min_date(df, min_date=None, max_date=None):
    if min_date is None and max_date is None:
        raise ValueError
    elif min_date and max_date is None:
        return df[df['Time'] >= min_date]
    elif min_date is None and max_date:
        return df[df['Time'] <= max_date]
    else:
        return df[(min_date <= df['Time']) & (df['Time'] <= max_date)]


def set_row_time_idx(x):
    time_idx = x.name
    return time_idx


def preprocess_single_df_fisherman(data, min_date, max_date):
    data = filter_df_by_min_date(data, min_date, max_date)
    data.drop_duplicates(inplace=True)
    sensor = data['Sensor'].iloc[0]
    data_3h = data.set_index('Time').resample('3H').mean()
    data_3h.fillna(method='bfill', inplace=True)
    data_3h['Sensor'] = sensor
    data_3h['Time'] = data_3h.index
    data_3h.reset_index(inplace=True, drop=True)
    data_3h['Minute'] = data_3h.Time.dt.minute.astype(str).astype("category")
    data_3h['Hour'] = data_3h.Time.dt.hour.astype(str).astype("category")
    data_3h['DayOfMonth'] = data_3h.Time.dt.day.astype(str).astype("category")
    data_3h['DayOfWeek'] = data_3h.Time.dt.weekday.astype(str).astype("category")
    data_3h['time_idx'] = data_3h.apply(lambda x: set_row_time_idx(x), axis=1)
    return data_3h


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
