import numpy as np
import pandas as pd
from config import DATETIME_COLUMN
from typing import List, Union, Dict


def filter_df_by_date(df, min_date=None, max_date=None):
    if min_date is None and max_date is None:
        raise ValueError
    elif min_date and max_date is None:
        return df[df[DATETIME_COLUMN] >= min_date]
    elif min_date is None and max_date:
        return df[df[DATETIME_COLUMN] <= max_date]
    else:
        return df[(min_date <= df[DATETIME_COLUMN]) & (df[DATETIME_COLUMN] <= max_date)]


def add_dt_columns(data: Union[pd.DataFrame, Dict], dt_attributes: List = []):
    if isinstance(data, pd.DataFrame):
        if 'minute' in dt_attributes:
            data['minute'] = data[DATETIME_COLUMN].dt.minute.astype(str).astype("category")
        if 'hour' in dt_attributes:
            data['hour'] = data[DATETIME_COLUMN].dt.hour.astype(str).astype("category")
        if 'day_of_month' in dt_attributes:
            data['day_of_month'] = data[DATETIME_COLUMN].dt.day.astype(str).astype("category")
        if 'day_of_week' in dt_attributes:
            data['day_of_week'] = data[DATETIME_COLUMN].dt.weekday.astype(str).astype("category")
        if 'month' in dt_attributes:
            data['month'] = data[DATETIME_COLUMN].dt.month.astype(str).astype("category")
        if 'year' in dt_attributes:
            data['year'] = data[DATETIME_COLUMN].dt.year.astype(str).astype("category")
        if 'is_weekend' in dt_attributes:
            if 'day_of_week' in dt_attributes:
                data['is_weekend'] = data.apply(lambda row: 1 if row['day_of_week'] in ['4', '5'] else 0, axis=1)
                data['is_weekend'] = data['is_weekend'].astype(str).astype("category")
            else:
                raise ValueError()

    elif isinstance(data, Dict):
        if 'hour' in dt_attributes:
            data['hour'] = data[DATETIME_COLUMN].hour
        if 'day_of_month' in dt_attributes:
            data['day_of_month'] = data[DATETIME_COLUMN].day
        if 'day_of_week' in dt_attributes:
            data['day_of_week'] = data[DATETIME_COLUMN].weekday()
        if 'month' in dt_attributes:
            data['month'] = data[DATETIME_COLUMN].month

    return data

def add_log_column(data, col_name):
    new_col_name = 'log_' + col_name
    data[new_col_name] = np.log(data[col_name])


def get_dataloader(ts_ds, is_train, config):
    dataloader = ts_ds.to_dataloader(train=is_train, batch_size=config["Train"].get("BatchSize"),
                                     num_workers=config["Train"].get("CPU"))
    return dataloader

