import os
import multiprocessing
import numpy as np

KEY_DELIMITER = '_'
DATETIME_COLUMN = "Date"

DATA = "Data"
DATA_BASE_FOLDER = os.path.join(DATA)

PLOT = "/tmp/noam/plots"
PLOT = os.path.join(PLOT)

STUDY = "Study"
STUDY_BASE_FOLDER = os.path.join(STUDY)


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
    study_reg_path = os.path.join(study_path, "Reg")
    study_class_path = os.path.join(study_path, "Class")

    plot_path = os.path.join(PLOT, dataset_name)

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.isdir(study_path):
        os.mkdir(study_path)
    if not os.path.isdir(study_reg_path):
        os.mkdir(study_reg_path)
    if not os.path.isdir(study_class_path):
        os.mkdir(study_class_path)
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
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman', 'Reg'),
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
            "TestDataFramePicklePath": os.path.join('pkl', 'fisherman2_test_df.pkl'),
            "ValDataFramePicklePath": os.path.join('pkl', 'fisherman2_val_df.pkl'),
            "GroupMappingPicklePath": os.path.join('pkl', 'fisherman2_group_mapping.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman2', 'Reg'),
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
            "Filename": "SyntheticDataset.csv",
            "TestDataFramePicklePath": os.path.join('pkl', 'synthetic_test_df.pkl'),
            "TestTsDsPicklePath": os.path.join('pkl', 'synthetic_test_ts_ds.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic', 'Reg'),
            "EncoderLength": 200,
            "PredictionLength": 7,
            "GroupKeyword": "Series",
            "ValueKeyword": "Value",
            "NumSeries": 10,
            "NumSubSeries": 20,
            "TimeStepsSubSeries": 24*7,
            "NumCorrelatedSeries": 5,
            "Noise": 0.5,
            "Trend": 1,
            "Level": 1,
        },
        "Stallion": {
            "EncoderLength": 14,
            "PredictionLength": 7,
        },
        "Straus": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'Straus'),
            "TestDataFramePicklePath": os.path.join('pkl', 'straus_test_df.pkl'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Straus', 'Reg'),
            "ProcessedDataPath": os.path.join('pkl', 'straus_processed_df.pkl'),
            "GroupKeyword": "key",
            "GroupColumns": ['PartId', 'OrderStepId', 'QmpId'],
            "ValueKeyword": "ActualValue",
            "ExceptionKeyword": "is_stoppage",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "EncoderLength": 360,
            "PredictionLength": 30,
        },
        "SMD": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'SMD'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'Straus', 'Reg'),
            "GroupColumns": ["Machine", "Dimension"],
            "ValueKeyword": "Value",
            "LabelKeyword": "Label",
            "EncoderLength": 300,
            "PredictionLength": 10,
        },
        "MSL": {
            "Path": os.path.join(DATA_BASE_FOLDER, 'SMAP_MSL'),
            "StudyRegPath": os.path.join(STUDY_BASE_FOLDER, 'MSL', 'Reg'),
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
                "Synthetic": {
                    "0": {
                        "lb": -5.5,
                        "hb": 4,
                    },
                    "1": {
                        "lb": -1.5,
                        "hb": 2.75,
                    },
                    "2": {
                        "lb": -0.8,
                        "hb": 0.6,
                    },
                    "3": {
                        "lb": -1.5,
                        "hb": 6.5,
                    },
                    "4": {
                        "lb": -4,
                        "hb": 6,
                    },
                    "5": {
                        "lb": -2,
                        "hb": 2.75,
                    },
                    "6": {
                        "lb": -2,
                        "hb": 8.6,
                    },
                    "7": {
                        "lb": -1.5,
                        "hb": 2,
                    },
                    "8": {
                        "lb": -1,
                        "hb": 1.5,
                    },
                    "9": {
                        "lb": -1.3,
                        "hb": 0.5,
                    },
                },
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
            "AlertMaxPredictionSteps": 6,
            "AlertMinPredictionSteps": 0,
            "RestartSteps": 0,
            "Rewards": {
                "MissedAlert": -100,
                "FalseAlert": -100,
                "GoodAlert": 10,
            }
        },
        "THTS": {
            "NumTrials": 75,
            "TrialLength": 8
        }

    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)

    return config
