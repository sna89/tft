import os
import pathlib
from Utils.utils import save_to_pickle, load_pickle

PKL_FOLDER = "pkl"
CONFIG_PKL_FILE = "config.pkl"

OBSERVED_KEYWORD = "Observed"
NOT_OBSERVED_KEYWORD = "NotObserved"

OBSERVED_LB_KEYWORD = "ObservedLB"
OBSERVED_UB_KEYWORD = "ObservedUB"

NOT_OBSERVED_LB_KEYWORD = "NotObservedLB"
NOT_OBSERVED_UB_KEYWORD = "NotObservedUB"

KEY_DELIMITER = '_'
DATETIME_COLUMN = "Date"

DATA = "Data"
DATA_BASE_FOLDER = os.path.join(DATA)

path = pathlib.Path(__file__).parent.resolve()
PLOT = os.path.join(path, "Plots")

RESULT = os.path.join(path, "Results")

STUDY = "Study"
STUDY_BASE_FOLDER = os.path.join(STUDY)

CLASSIFICATION_TASK_TYPE = "class"
CLASSIFICATION_FOLDER = "Class"

REGRESSION_TASK_TYPE = "reg"
REGRESSION_FOLDER = "Reg"

THTS_TASK_TYPE = "thts"
THTS_FOLDER = "THTS_BugFix_Deterministic"

COMBINED_TASK_TYPE = "combined"
COMBINED_FOLDER = "Combined"

ROLLOUT_TASK_TYPE = "rollout"

QUANTILES = [0.01, 0.2, 0.5, 0.8, 0.99]


def get_task_folder_name(task_type=REGRESSION_TASK_TYPE, prefix=None):
    folder_name = None
    if task_type == REGRESSION_TASK_TYPE:
        folder_name = prefix + "_" + REGRESSION_FOLDER if prefix else REGRESSION_FOLDER
    elif task_type == CLASSIFICATION_TASK_TYPE:
        folder_name = prefix + "_" + CLASSIFICATION_FOLDER if prefix else CLASSIFICATION_FOLDER
    elif task_type == THTS_TASK_TYPE:
        folder_name = prefix + "_" + THTS_FOLDER if prefix else THTS_FOLDER
    return folder_name


def init_base_folders():
    if not os.path.isdir(DATA):
        os.mkdir(DATA)
    if not os.path.isdir(STUDY):
        os.mkdir(STUDY)
    if not os.path.isdir(PLOT):
        os.mkdir(PLOT)
    if not os.path.isdir(PKL_FOLDER):
        os.mkdir(PKL_FOLDER)


def init_study_folders(dataset_name, model_name):
    study_path = os.path.join(STUDY, dataset_name)
    study_reg_path = os.path.join(study_path, REGRESSION_FOLDER)
    study_class_path = os.path.join(study_path, CLASSIFICATION_FOLDER)
    study_combined_path = os.path.join(study_path, COMBINED_FOLDER)

    if not os.path.isdir(study_path):
        os.mkdir(study_path)

    if not os.path.isdir(study_reg_path):
        os.mkdir(study_reg_path)

    study_reg_model_path = os.path.join(study_reg_path, model_name)
    if not os.path.isdir(study_reg_model_path):
        os.mkdir(study_reg_model_path)

    if not os.path.isdir(study_class_path):
        os.mkdir(study_class_path)

    study_class_model_path = os.path.join(study_class_path, model_name)
    if not os.path.isdir(study_class_model_path):
        os.mkdir(study_class_model_path)

    if not os.path.isdir(study_combined_path):
        os.mkdir(study_combined_path)

    study_combined_model_path = os.path.join(study_combined_path, model_name)
    if not os.path.isdir(study_combined_model_path):
        os.mkdir(study_combined_model_path)


def init_dataset_folders(dataset_name, model_name):
    dataset_path = os.path.join(DATA, dataset_name)
    plot_path = os.path.join(PLOT, dataset_name)

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    init_study_folders(dataset_name, model_name)


def get_config(dataset_name, model_name):
    init_base_folders()
    init_dataset_folders(dataset_name, model_name)

    dataset_config = {
        "Electricity": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Electricity', 'LD2011_2014.txt'),
            "EncoderLength": 168,
            "PredictionLength": 24,
            "NumGroups": 11,
            "ProcessedDfColumnNames": ['date', 'group', 'value'],
            "StartDate": "2012-01-01",
            "EndDate": "2013-01-01"
        },
        "Fisherman": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Fisherman'),
            "TestDataFramePicklePath": os.path.join(PKL_FOLDER, 'fisherman_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join(PKL_FOLDER, 'fisherman_val_df.pkl'),
            "GroupMappingPicklePath": os.path.join(PKL_FOLDER, 'fisherman_group_mapping.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman', REGRESSION_FOLDER),
            "EncoderLength": 144,
            "PredictionLength": 6,
            "GroupKeyword": "Sensor",
            "ValueKeyword": "Value",
            "ExceptionKeyword": "future_exceed",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "Resample": True,
        },
        "Fisherman2": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Fisherman2'),
            "TestDataFramePicklePath": os.path.join(PKL_FOLDER, 'fisherman2_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join(PKL_FOLDER, 'fisherman2_val_df.pkl'),
            "GroupMappingPicklePath": os.path.join(PKL_FOLDER, 'fisherman2_group_mapping.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman2', REGRESSION_FOLDER),
            "EncoderLength": 72,
            "PredictionLength": 6,
            "GroupKeyword": "Key",
            "ValueKeyword": "Value",
            "ExceptionKeyword": "future_exceed",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "Resample": False,
            "SlidingWindow": False,
            "SlidingWindowSamples": 24,
        },
        "Synthetic": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Synthetic'),
            "Filename": "SyntheticData_Version_15_Correlated_10.csv",
            "TestDataFramePicklePath": os.path.join(PKL_FOLDER, 'synthetic_test_df.pkl'),
            "TrainDataFramePicklePath": os.path.join(PKL_FOLDER, 'synthetic_train_df.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic', REGRESSION_FOLDER, model_name),
            "StudyClassPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic', CLASSIFICATION_FOLDER, model_name),
            "StudyCombinedPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic', COMBINED_FOLDER, model_name),
            "EncoderLength": 400,
            "PredictionLength": 14,
            "GroupKeyword": "Series",
            "ValueKeyword": "Value",
            "ObservedBoundKeyword": "ObservedBound",
            "UnobservedBoundKeyword": "UnObservedBound",
        },
        "Stallion": {
            "EncoderLength": 14,
            "PredictionLength": 7,
        },
        "Straus": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Straus'),
            "TestDataFramePicklePath": os.path.join(PKL_FOLDER, 'straus_test_df.pkl'),
            "TrainDataFramePicklePath": os.path.join(PKL_FOLDER, 'straus_train_df.pkl'),
            "TestTsDsPicklePath": os.path.join(PKL_FOLDER, 'straus_test_ts_ds.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Straus', REGRESSION_FOLDER, model_name),
            "GroupKeyword": "QmpId",
            "GroupColumns": ['PartId', 'OrderStepId', "QmpId"],
            "ValueKeyword": "ActualValue",
            # "ValueKeyword": "ShellIndex",
            "ExceptionKeyword": "is_stoppage",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "EncoderLength": 360,
            "PredictionLength": 30,
            "ObservedBoundKeyword": "ObservedBound",
        },
        "SMD": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'SMD'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'SMD', REGRESSION_FOLDER),
            "GroupColumns": ["Machine", "Dimension"],
            "ValueKeyword": "Value",
            "LabelKeyword": "Label",
            "EncoderLength": 300,
            "PredictionLength": 10,
        },
        "MSL": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'SMAP_MSL'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'MSL', REGRESSION_FOLDER),
            "GroupKeyword": "Channel",
            "ValueKeyword": "Value",
            "LabelKeyword": "Label",
            "EncoderLength": 100,
            "PredictionLength": 10,
        }
    }

    train_config = {
        "Train": {
            "BatchSize": 32,
            "CPU": 0,
            "Synthetic": {
                "TrainRatio": 0.7,
                "ValRatio": 0.2
            },
            "Straus": {
                "TrainRatio": 0.6,
                "ValRatio": 0.2
            }
        }
    }

    anomaly_config = {
        "AnomalyConfig":
            {
                "Synthetic": {},
                "Fisherman": {},
                "Straus": {},
            },
        "Env": {
            "ConsecutiveExceptions": 1,
            "AlertMaxPredictionSteps": 4,
            "RestartSteps": 10,
            "Rewards": {
                "CheapFP": {
                    "MissedAlert": -1000,
                    "FalseAlert": -10,
                    "GoodAlert": 50,
                },
                "ExpensiveFP": {
                    "MissedAlert": -1000,
                    "FalseAlert": -100,
                    "GoodAlert": 50,
                }
            }
        },
        "THTS": {
            "NumTrials": 30,
            "TrialLength": 6
        }

    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)
    update_config(config)
    return config


def load_config(path):
    return load_pickle(os.path.join(path, PKL_FOLDER, CONFIG_PKL_FILE))


def update_config(config):
    save_to_pickle(config, os.path.join(PKL_FOLDER, CONFIG_PKL_FILE))


def get_env_steps_from_alert(config):
    return config.get("Env").get("AlertMaxPredictionSteps")


def get_restart_steps(config):
    return config.get("Env").get("RestartSteps")


def get_env_restart_steps(config):
    return get_restart_steps(config) + 1


def get_false_alert_reward(config):
    return config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("FalseAlert")


def get_missed_alert_reward(config):
    return config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("MissedAlert")


def get_good_alert_reward(config):
    return config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("GoodAlert")


def get_tree_depth(config):
    return config.get("THTS").get("TrialLength")


def get_num_quantiles():
    return len(QUANTILES)
