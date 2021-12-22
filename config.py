import os
from utils import save_to_pickle, load_pickle

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

PLOT = "Plots"
PLOT = os.path.join(PLOT)

STUDY = "Study"
STUDY_BASE_FOLDER = os.path.join(STUDY)

CLASSIFICATION_TASK_TYPE = "class"
CLASSIFICATION_FOLDER = "Class"

REGRESSION_TASK_TYPE = "reg"
REGRESSION_FOLDER = "Reg"

COMBINED_TASK_TYPE = "combined"
COMBINED_FOLDER = "Combined"

ROLLOUT_TASK_TYPE = "rollout"


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
            "Filename": "SyntheticData_Version_0_Correlated_0.csv",
            "TestDataFramePicklePath": os.path.join(PKL_FOLDER, 'synthetic_test_df.pkl'),
            "TestTsDsPicklePath": os.path.join(PKL_FOLDER, 'synthetic_test_ts_ds.pkl'),
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
            "TestTsDsPicklePath": os.path.join(PKL_FOLDER, 'straus_test_ts_ds.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Straus', REGRESSION_FOLDER, model_name),
            "GroupKeyword": "key",
            "GroupColumns": ['PartId', 'OrderStepId'],
            # "ValueKeyword": "ActualValue",
            "ValueKeyword": "ShellIndex",
            "ExceptionKeyword": "is_stoppage",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "EncoderLength": 360,
            "PredictionLength": 30,
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
            "TrainRatio": 0.6,
            "ValRatio": 0.2,
            "CPU": 0
        }
    }

    anomaly_config = {
        "AnomalyConfig":
            {
                "Synthetic": {},
                "Fisherman": {
                    "U100330": {
                        "lb": -16.7225,
                        "hb": -9.5,
                    },
                    "U100314": {
                        "lb": -19.15,
                        "hb": 20.9,
                    },
                    "U100329": {
                        "lb": 2.87,
                        "hb": 18.8,
                    },
                    "U100337": {
                        "lb": 0.5,
                        "hb": 6.5,
                    },
                    "U106724": {
                        "lb": -0.65,
                        "hb": 7,
                    },
                    "U100312": {
                        "lb": 0.5,
                        "hb": 6.5,
                    },
                    "U100309": {
                        "lb": 0.5,
                        "hb": 5.5,
                    },
                    "U100310": {
                        "lb": 5.1,
                        "hb": 14,
                    },
                    "U106755": {
                        "lb": 15,
                        "hb": 25,
                    },
                }

            },
        "Env": {
            "ConsecutiveExceptions": 1,
            "AlertMaxPredictionSteps": 1,
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
            "NumTrials": 100,
            "TrialLength": 3
        }

    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)
    update_config(config)
    return config


def load_config():
    return load_pickle(os.path.join(PKL_FOLDER, CONFIG_PKL_FILE))


def update_config(config):
    save_to_pickle(config, os.path.join(PKL_FOLDER, CONFIG_PKL_FILE))