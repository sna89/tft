import concurrent.futures
import pandas as pd
from EnvCommon.env_thts_common import get_group_names_from_df
from config import get_config
from DataBuilders.build import split_df
from Tasks.thts_task import run_thts_task_for_group
from Tasks.rollout_task import run_rollout_task
from Tasks.time_series_task import run_time_series_task
from DataBuilders.build import build_data, process_data, update_bounds
import warnings
import os
from plot import plot_data
from Algorithms.anomaly_detection.anomaly_detection import AnomalyDetection
from config import REGRESSION_TASK_TYPE, CLASSIFICATION_TASK_TYPE
from Utils.utils import save_to_pickle, create_chunks

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")
    model_name = os.getenv("MODEL_NAME_REG")

    config = get_config(dataset_name, model_name)

    data = build_data(config, dataset_name)
    data = process_data(config, dataset_name, data)

    train_df, val_df, test_df = split_df(config, dataset_name, data)
    update_bounds(config, dataset_name, train_df, val_df, test_df)
    # plot_data(config, dataset_name, train_df)
    # plot_data(config, dataset_name, val_df)
    # plot_data(config, dataset_name, test_df)

    fitted_reg_model = None

    if os.getenv("REG_TASK") == "True":
        fitted_reg_model = run_time_series_task(config,
                                                REGRESSION_TASK_TYPE,
                                                dataset_name,
                                                train_df,
                                                val_df,
                                                test_df,
                                                evaluate=True,
                                                plot=False)

    if os.getenv("CLASS_TASK") == "True":
        run_time_series_task(config,
                             CLASSIFICATION_TASK_TYPE,
                             dataset_name,
                             train_df,
                             val_df,
                             test_df,
                             evaluate=False)

    if os.getenv("ROLLOUT_REG_TASK") == "True":
        run_rollout_task(config,
                         dataset_name,
                         test_df,
                         REGRESSION_TASK_TYPE)

    if os.getenv("ROLLOUT_CLASS_TASK") == "True":
        run_rollout_task(config,
                         dataset_name,
                         test_df,
                         CLASSIFICATION_TASK_TYPE)

    if os.getenv("THTS_TASK") == "True":
        save_to_pickle(test_df, config.get("TestDataFramePicklePath"))
        save_to_pickle(train_df, config.get("TrainDataFramePicklePath"))

        group_names = get_group_names_from_df(config, test_df)

        # if False:
        # PARALLEL_GROUPS = 10
        # for group_names_chunk in create_chunks(group_names, PARALLEL_GROUPS):
        #     with concurrent.futures.ProcessPoolExecutor() as executor:
        #         executor.map(run_thts_task_for_group, group_names_chunk)
        # else:
        run_thts_task_for_group(group_names[0])

    if os.getenv("APPLY_ANOMALY_DETECTION") == "True":
        detector = AnomalyDetection(config, os.getenv("MODEL_NAME_REG"), dataset_name, fitted_reg_model)
        anomaly_df = detector.detect_and_evaluate(train_df, False)

    # trajectory_sample = TrajectorySample(ad_env, config, fitted_reg_model, val_df, test_df, num_trajectories=5000)
    # trajectory_sample.run()
