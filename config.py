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
            "StudyPath": os.path.join(STUDY_BASE_FOLDER, 'Fisherman'),
            "EncoderLength": 56,
            "PredictionLength": 7
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
            "Timesteps": 600
        },
        "Stallion": {
            "EncoderLength": 14,
            "PredictionLength": 7,
        },
    }

    train_config = {
        "Train": {
            "BatchSize": 128,
            "TrainRatio": 0.6,
            "ValRatio": 0.2,
            "CPU": 0
        }
    }

    anomaly_config = {
        "AnomalyConfig":
            {
                "series_0": {
                    "lb": 1,
                    "hb": 2.3,
                },
                "series_1": {
                    "lb": -0.5,
                    "hb": 0.5,
                },
                "series_2": {
                    "lb": -1.5,
                    "hb": 0.5,
                },
                # "series_3": {
                #     "lb": -1,
                #     "hb": 5,
                # },
                # "series_4": {
                #     "lb": -1,
                #     "hb": 2,
                # },
                # "series_5": {
                #     "lb": -1,
                #     "hb": 1,
                # },
                # "series_6": {
                #     "lb": -0.5,
                #     "hb": 0.5,
                # },
                # "series_7": {
                #     "lb": -0.1,
                #     "hb": 0.1,
                # },
                # "series_8": {
                #     "lb": -0.5,
                #     "hb": 0.5,
                # },
                # "series_9": {
                #     "lb": -0.3,
                #     "hb": 1,
                # }
            },
        "Env": {
            "AlertMaxPredictionSteps": 4,
            "AlertMinPredictionSteps": 0,
            "RestartSteps": 5,
            "Rewards": {
                "MissedAlert": -1000,
                "FalseAlert": -100,
                "GoodAlert": 10,
            }
        },
        "THTS": {
            "NumTrials": 200,
            "TrialLength": 9,
            "UCTBias": np.sqrt(2),
            "Runs": 1
        }

    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)

    return config
