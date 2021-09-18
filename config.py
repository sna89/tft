import numpy as np
from utils import PKL
import os

DATETIME_COLUMN = "Date"


def get_config():
    config = {
        "model": "TFT",
        # "raw_data_path": "/home/sna89/pycharm_project_99/Data/Fisherman/",
        # "save_data_path": "/home/sna89/pycharm_project_99/Data/Fisherman/processed_data.pkl",
        "load_data_path": "/home/sna89/pycharm_project_99/Data/Fisherman/processed_data.pkl",
        "study": False,
        # "save_study_path": "/home/sna89/pycharm_project_99/Study/study_0.pkl",
        "load_study_path": "/home/sna89/pycharm_project_99/Study/study_0.pkl",
        "train": False,
        "load_model_path": "/home/sna89/pycharm_project_99/tb_logs/my_model/version_3/checkpoints/epoch=5-step=5279"
                           ".ckpt",
        "val_df_pkl_path": os.path.join(PKL, 'val_df.pkl'),
        "test_df_pkl_path": os.path.join(PKL, 'test_df.pkl'),

        "plot_data": False,
        "plot_predictions": False,
        "Data": {
            "EncoderLength": 56,
            "PredictionLength": 7,
            "GroupKeyword": "Sensor",
            "ValueKeyword": "Value",
            "DatetimeAdditionalColumns": ['hour', 'day_of_month', 'day_of_week', 'minute'],
            "ResampleFreq": "1H"
        },
        "DataLoader":  {
            "BatchSize": 32,
            "TrainRatio": 0.6,
            "ValRatio": 0.2,
            "CPU": 0
        },
        "Trainer": {
            "patience": 3,
            "delta": 1e-4,
            "gpus": 1,
            "epochs": 50
        },
        "Env": {
            "AlertMaxPredictionSteps": 4,
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
            "UCTBias": np.sqrt(2)
        },
        "Anomaly": {
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
                "lb": -3,
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
    }

    return config
