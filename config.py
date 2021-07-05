import os
import multiprocessing

BASE_FOLDER = os.path.join('Data')
STUDY_BASE_FOLDER = os.path.join('Study')


def get_config(dataset_name):
    dataset_config = {
        "Electricity": {
            "Path": os.path.join(BASE_FOLDER, 'Electricity', 'LD2011_2014.txt'),
            "EncoderLength": 168,
            "PredictionLength": 24,
            "NumGroups": 11,
            "ProcessedDfColumnNames": ['date', 'group', 'value'],
            "StartDate": "2012-01-01",
            "EndDate": "2013-01-01"
        },
        "Fisherman": {
            "Path": os.path.join(BASE_FOLDER, 'Fisherman'),
            "EncoderLength": 100,
            "PredictionLength": 20
        },
        "Synthetic": {
            "Path": os.path.join(BASE_FOLDER, 'Synthetic'),
            "StudyPath": os.path.join(STUDY_BASE_FOLDER, 'Synthetic'),
            "EncoderLength": 30,
            "PredictionLength": 10,
            "Series": 10,
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
                    "lb": -0.5,
                    "hb": 0.5,
                },
                "series_1": {
                    "lb": -0.3,
                    "hb": 0.3,
                },
                "series_2": {
                    "lb": -1.5,
                    "hb": 0.5,
                },
                "series_3": {
                    "lb": -1,
                    "hb": 5,
                },
                "series_4": {
                    "lb": -1,
                    "hb": 2,
                },
                "series_5": {
                    "lb": -1,
                    "hb": 1,
                },
                "series_6": {
                    "lb": -0.5,
                    "hb": 0.5,
                },
                "series_7": {
                    "lb": -0.1,
                    "hb": 0.1,
                },
                "series_8": {
                    "lb": -0.5,
                    "hb": 0.5,
                },
                "series_9": {
                    "lb": -0.3,
                    "hb": 1,
                }
            }
    }

    config = dict(dict(dataset_config[dataset_name], **train_config), **anomaly_config)

    return config
