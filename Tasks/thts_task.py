from DataBuilders.build import convert_df_to_ts_data
from utils import save_to_pickle
import gym
from Algorithms.thts.max_uct import MaxUCT


def run_thts_task(config, dataset_name, train_df, test_df):
    save_to_pickle(test_df, config.get("TestDataFramePicklePath"))

    _, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, "reg")
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, "reg")
    save_to_pickle(test_ts_ds, config.get("TestTsDsPicklePath"))

    ad_env = gym.make("gym_ad:ad-v0")
    thts = MaxUCT(ad_env, config)
    thts.run(test_df)
