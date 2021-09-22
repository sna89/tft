import numpy as np
import pandas as pd
from config import DATETIME_COLUMN
from typing import List, Union, Dict
import os
import torch


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
        if 'second' in dt_attributes:
            data['second'] = data[DATETIME_COLUMN].dt.second.astype(str).astype("category")
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


def get_group_indices_mapping(config, dl):
    mapping = dl.dataset.decoded_index.groupby(config.get("GroupKeyword")).indices
    return mapping


def reverse_key_value_mapping(d):
    return {v: k for k, v in d.items()}


def assign_time_idx(df, dt_col):
    dt_data = pd.to_datetime(pd.unique(df[dt_col].sort_values()))
    dt_time_idx_mapping = dict(zip(pd.to_datetime(dt_data), list(range(1, len(dt_data) + 1))))
    df['time_idx'] = [dt_time_idx_mapping[pd.to_datetime(dt)] for dt in df[dt_col]]
    return df


def get_group_lower_and_upper_bounds(config, group_name):
    bounds = config.get("AnomalyConfig").get(os.getenv("DATASET")).get(group_name)
    lb, ub = bounds.values()
    return lb, ub


def add_future_exceed(config, data, group):
    prediction_len = config.get("PredictionLength")
    lb, ub = get_group_lower_and_upper_bounds(config, group)
    exception_col = config.get("ExceptionKeyword")
    data[exception_col] = is_future_exceed(config, data, prediction_len, lb, ub)
    data[exception_col] = data[exception_col].astype(int).astype(str).astype("category")
    return data


def is_future_exceed(config, data, prediction_len, lb, ub):
    shifted_values = data.shift(-prediction_len)[config.get("ValueKeyword")]
    return (shifted_values < lb) | (shifted_values > ub)


def get_group_idx_mapping(config, model, df):
    if isinstance(model.hparams.embedding_labels, dict) and \
            config.get("GroupKeyword") in model.hparams.embedding_labels:
        return model.hparams.embedding_labels[config.get("GroupKeyword")]
    else:
        group_name_list = list(df[config.get("GroupKeyword")].unique())
        return {group_name: group_name for group_name in group_name_list}


def is_group_prediction_out_of_bound(group_prediction, lb, ub):
    out_of_bound = torch.where((group_prediction < lb) | (group_prediction > ub), 1, 0)
    if sum(out_of_bound) > 0:
        idx = (out_of_bound == 1).nonzero()[0].item()
        return True, idx
    else:
        return False, -1
