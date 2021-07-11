import pickle
from typing import List
import random
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd


def save_to_pickle(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def get_argmax_from_list(l: List, choose_random=True):
    max_value = max(l)
    max_indices = [i for i, value in enumerate(l) if value == max_value]
    if choose_random:
        max_idx = random.choice(max_indices)
        return max_idx
    return max_indices


def set_env_to_state(env, state):
    env.current_state = state





# def get_last_sample_df_from_ts_ds(ts_ds: TimeSeriesDataSet, df: pd.DataFrame()):
#     last_sample_time_idx_last = ts_ds.decoded_index.time_idx_last.max()
#     decoded_index = ts_ds.decoded_index
#     last_sample_time_idx_first = int(decoded_index[decoded_index.time_idx_last == last_sample_time_idx_last]['time_idx_first'].unique())
#     last_sample_df = df[df.time_idx.isin(list(range(last_sample_time_idx_first, last_sample_time_idx_last + 1)))]
#     return last_sample_df