import numpy as np
import pandas as pd
from config import DATETIME_COLUMN, OBSERVED_KEYWORD, NOT_OBSERVED_KEYWORD
from typing import List, Union, Dict
import os
import torch


def filter_df_by_date(df, min_date=None, max_date=None):
    if min_date is None and max_date is None:
        raise ValueError
    elif min_date and max_date is None:
        return df[df[DATETIME_COLUMN] > min_date]
    elif min_date is None and max_date:
        return df[df[DATETIME_COLUMN] < max_date]
    else:
        return df[(min_date < df[DATETIME_COLUMN]) & (df[DATETIME_COLUMN] < max_date)]


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


def get_dataloader(ts_ds, is_train, config, drop_last=False):
    dataloader = ts_ds.to_dataloader(train=is_train,
                                     batch_size=config["Train"].get("BatchSize"),
                                     num_workers=config["Train"].get("CPU"),
                                     drop_last=drop_last)
    return dataloader


def get_idx_list(dl, x, step):
    index_df = dl.dataset.x_to_index(x)
    idx_list = list(range(index_df.index.min(), index_df.index.max(), step))
    return idx_list


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
    is_observed = os.getenv("IS_EXCEPTION_OBSERVED") == "True"
    if is_observed:
        bounds = config.get("AnomalyConfig").get(os.getenv("DATASET")).get(group_name).get(OBSERVED_KEYWORD)
    else:
        bounds = config.get("AnomalyConfig").get(os.getenv("DATASET")).get(group_name).get(NOT_OBSERVED_KEYWORD)
    lb, ub = bounds.values()
    return lb, ub


def create_bounds_labels(config, data):
    is_observed = os.getenv("IS_EXCEPTION_OBSERVED") == "True"
    bound_col = config.get("ObservedBoundKeyword") if is_observed else config.get("UnobservedBoundKeyword")

    if is_observed:
        data = add_observed_bound_column(bound_col, data)
    else:
        data = add_unobserved_bound_column(bound_col, data)

    groups = pd.unique(data[config.get("GroupKeyword")])
    for group in groups:
        group_data = data[data[config.get("GroupKeyword")] == group]
        horizon = config.get("PredictionLength")
        data.loc[group_data.index, bound_col] = add_bounds_label(config, group_data, group, horizon)

    data[bound_col] = data[bound_col].astype(int).astype(str).astype("category")
    return data


def add_observed_bound_column(observed_bound_col, data):
    data[observed_bound_col] = False
    return data


def add_unobserved_bound_column(unobserved_bound_col, data):
    data[unobserved_bound_col] = False
    return data


def add_bounds_label(config, data, group, horizon):
    lb, ub = get_group_lower_and_upper_bounds(config, group)
    shifted_values = data.shift(-horizon)[config.get("ValueKeyword")]
    data["Exception"] = (shifted_values < lb) | (shifted_values > ub)
    data["ConsecutiveExceptions"] = data["Exception"].rolling(min_periods=1,
                                                              window=config.get("Env").get("ConsecutiveExceptions")).sum()
    return data["ConsecutiveExceptions"] == config.get("Env").get("ConsecutiveExceptions")


def get_group_id_group_name_mapping(config, ts_ds):
    group_id_group_name_mapping = {}
    group_ids = pd.unique(ts_ds.index["group_id"])
    ts_ds_index = ts_ds.index.reset_index()

    for group_id in group_ids:
        group_indices = ts_ds_index[ts_ds_index["group_id"] == group_id].index
        group_name = ts_ds.decoded_index.iloc[group_indices.min()][config.get("GroupKeyword")]
        group_id_group_name_mapping[group_id] = group_name

    return group_id_group_name_mapping


def is_group_prediction_out_of_bound(group_prediction, lb, ub):
    out_of_bound = torch.where((group_prediction < lb) | (group_prediction > ub), 1, 0)
    if sum(out_of_bound) > 0:
        idx = (out_of_bound == 1).nonzero()[0].item()
        return True, idx
    else:
        return False, -1




