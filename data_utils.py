import numpy as np
import pandas as pd


def filter_df_by_date(df, min_date=None, max_date=None):
    if min_date is None and max_date is None:
        raise ValueError
    elif min_date and max_date is None:
        return df[df['Time'] >= min_date]
    elif min_date is None and max_date:
        return df[df['Time'] <= max_date]
    else:
        return df[(min_date <= df['Time']) & (df['Time'] <= max_date)]


def add_dt_columns(data, dt_attributes=[]):
    if 'minute' in dt_attributes:
        data['minute'] = data.date.dt.minute.astype(str).astype("category")
    if 'hour' in dt_attributes:
        data['hour'] = data.date.dt.hour.astype(str).astype("category")
    if 'day_of_month' in dt_attributes:
        data['day_of_month'] = data.date.dt.day.astype(str).astype("category")
    if 'day_of_week' in dt_attributes:
        data['day_of_week'] = data.date.dt.weekday.astype(str).astype("category")
    if 'month' in dt_attributes:
        data['month'] = data.date.dt.month.astype(str).astype("category")
    if 'year' in dt_attributes:
        data['year'] = data.date.dt.year.astype(str).astype("category")
    if 'is_weekend' in dt_attributes:
        if 'day_of_week' in dt_attributes:
            data['is_weekend'] = data.apply(lambda row: 1 if row['day_of_week'] in ['4', '5'] else 0, axis=1)
            data['is_weekend'] = data['is_weekend'].astype(str).astype("category")
        else:
            raise ValueError()
    return data


def add_log_column(data, col_name):
    new_col_name = 'log_' + col_name
    data[new_col_name] = np.log(data[col_name])



