import os
import multiprocessing
import numpy as np

DATETIME_COLUMN = "Date"

DATA = "Data"
STUDY = "Study"
PLOT = "Plots"
DATA_BASE_FOLDER = os.path.join(DATA)
STUDY_BASE_FOLDER = os.path.join(STUDY)
PLOT = os.path.join(PLOT)


def init_base_folders():
    if not os.path.isdir(DATA):
        os.mkdir(DATA)
    if not os.path.isdir(STUDY):
        os.mkdir(STUDY)
    if not os.path.isdir(PLOT):
        os.mkdir(PLOT)
    if not os.path.isdir('pkl'):
        os.mkdir('pkl')


def init_dataset_folders(dataset_name):
    dataset_path = os.path.join(DATA, dataset_name)
    study_path = os.path.join(STUDY, dataset_name)
    plot_path = os.path.join(PLOT, dataset_name)

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.isdir(study_path):
        os.mkdir(study_path)
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)


def get_config(dataset_name):
    init_base_folders()
    init_dataset_folders(dataset_name)

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
            "TestDataFramePicklePath": os.path.join('pkl', 'fisherman_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join('pkl', 'fisherman_val_df.pkl'),
            "GroupMappingPicklePath": os.path.join('pkl', 'fisherman_group_mapping.pkl'),
            "StudyPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman'),
            "EncoderLength": 56,
            "PredictionLength": 7,
            "GroupKeyword": "Sensor",
            "ValueKeyword": "Value",
            "ExceptionKeyword": "future_exceed",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "Resample": "False"
        },
        "Synthetic": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Synthetic'),
            "TestDataFramePicklePath": os.path.join('pkl', 'synthetic_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join('pkl', 'synthetic_val_df.pkl'),
            "StudyPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic'),
            "EncoderLength": 30,
            "PredictionLength": 1,
            "Series": 3,
            "Seasonality": 30,
            "Trend": 2,
            "Noise": 0.05,
            "Timesteps": 600,
            "GroupKeyword": "series",
            "ValueKeyword": "value",
            "DatetimeAdditionalColumns": ['month', 'day_of_month']
        },
        "Stallion": {
            "EncoderLength": 14,
            "PredictionLength": 7,
        },
        "Straus": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Straus'),
            "TestDataFramePicklePath": os.path.join('pkl', 'straus_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join('pkl', 'straus_val_df.pkl'),
            "StudyPath": os.path.join(STUDY_BASE_FOLDER, 'Straus'),
            "ProcessedDataPath": os.path.join('pkl', 'straus_processed_df.pkl'),
            "GroupKeyword": "key",
            "GroupColumns": ['PartId', 'OrderStepId', 'QmpId'],
            "ValueKeyword": "ActualValue",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute', 'second'],
            "EncoderLength": 180,
            "PredictionLength": 30,
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
                "Synthetic": {
                    "0": {
                        "lb": 1,
                        "hb": 2.3,
                    },
                    "1": {
                        "lb": -0.5,
                        "hb": 0.5,
                    },
                    "2": {
                        "lb": -1.5,
                        "hb": 0.5,
                    }
                },
                "Fisherman": {
                    "U100330": {
                        "lb": -22,
                        "hb": -8,
                    },
                    "U100314": {
                        "lb": -20,
                        "hb": 22,
                    },
                    "U100329": {
                        "lb": 0,
                        "hb": 11,
                    },
                    "U100337": {
                        "lb": -2,
                        "hb": 11,
                    },
                    "U106724": {
                        "lb": -1,
                        "hb": 6,
                    },
                    "U100312": {
                        "lb": 0,
                        "hb": 4.5,
                    },
                    "U100309": {
                        "lb": -5,
                        "hb": 12,
                    },
                    "U100310": {
                        "lb": 4,
                        "hb": 17,
                    },
                    "U106755": {
                        "lb": 13,
                        "hb": 28,
                    },
                }

            },
        "Env": {
            "AlertMaxPredictionSteps": 7,
            "AlertMinPredictionSteps": 0,
            "RestartSteps": 3,
            "Rewards": {
                "MissedAlert": -100,
                "FalseAlert": -100,
                "GoodAlert": 10,
            }
        },
        "THTS": {
            "NumTrials": 75,
            "TrialLength": 6,
            "UCTBias": np.sqrt(2),
            "Runs": 1
        }

    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)

    return config
