import os
from pathlib import Path
from DataBuilders.build import convert_df_to_ts_data
from EnvCommon.env_thts_common import get_group_names_from_df
from EnvCommon.predictor import Predictor
from config import load_config
from gym_ad.ad_env import AdEnv
from Utils.utils import get_model_from_checkpoint, load_pickle, create_chunks
from Algorithms.thts.max_uct import MaxUCT
import concurrent.futures
import gc


def run_thts_task_for_group(group_name):
    print("Run THTS task on group name: {}".format(group_name))
    path = Path(os.getcwd())

    config = load_config(path)
    forecasting_model = get_model_from_checkpoint(os.getenv("CHECKPOINT_REG"), os.getenv("MODEL_NAME_REG"))
    train_df = load_pickle(os.path.join(path, config.get("TrainDataFramePicklePath")))
    test_df = load_pickle(os.path.join(path, config.get("TestDataFramePicklePath")))

    dataset_name = os.getenv("DATASET")
    _, parameters = convert_df_to_ts_data(config, dataset_name, train_df, None, "reg")
    test_ts_ds, _ = convert_df_to_ts_data(config, dataset_name, test_df, parameters, "reg")

    env = AdEnv(config, forecasting_model, test_df, test_ts_ds)
    predictor = Predictor(config, forecasting_model, test_df, test_ts_ds)

    thts = MaxUCT(config, env, predictor, group_name)
    print("Start running THTS task on group name: {}".format(group_name))
    thts.run(test_df)
    del thts
    gc.collect()


if __name__ == "__main__":
    config = load_config()

    train_df = load_pickle(os.path.join(Path(os.getcwd()).parent.absolute(), config.get("TrainDataFramePicklePath")))
    group_names = get_group_names_from_df(config, train_df)

    PARALLEL_GROUPS = 10
    for group_names_chunk in create_chunks(group_names, PARALLEL_GROUPS):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(run_thts_task_for_group, group_names_chunk)
