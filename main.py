import pandas as pd
from config import get_config
from DataBuilders.build import split_df
from Tasks.thts_task import run_thts_task
from Tasks.rollout_task import run_rollout_task
from Tasks.time_series_task import run_time_series_task
from DataBuilders.build import build_data, process_data
import warnings
import os
from plot import plot_data
from Algorithms.trajectory_sample.trajectory_sample import TrajectorySample
from Algorithms.anomaly_detection.anomaly_detection import AnomalyDetection
from config import REGRESSION_TASK_TYPE, CLASSIFICATION_TASK_TYPE, COMBINED_TASK_TYPE

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    dataset_name = os.getenv("DATASET")

    config = get_config(dataset_name)

    data = build_data(config, dataset_name)
    data = process_data(config, dataset_name, data)
    # plot_data(config, dataset_name, data)

    train_df, val_df, test_df = split_df(config, dataset_name, data)
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
                             evaluate=True)

    if os.getenv("ROLLOUT_TASK") == "True":
        run_rollout_task(config,
                         dataset_name,
                         train_df,
                         test_df)

    # run_thts_task(config,
    #               dataset_name,
    #               train_df,
    #               test_df)

    if os.getenv("APPLY_ANOMALY_DETECTION") == "True":
        detector = AnomalyDetection(config, os.getenv("MODEL_NAME_REG"), dataset_name, fitted_reg_model)
        anomaly_df = detector.detect_and_evaluate(train_df, test_df, True)

    # trajectory_sample = TrajectorySample(ad_env, config, fitted_reg_model, val_df, test_df, num_trajectories=5000)
    # trajectory_sample.run()
