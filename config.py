import os
import multiprocessing

BASE_FOLDER = os.path.join('Data')


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
            "EncoderLength": 100,
            "PredictionLength": 50,
            "Series": 1,
            "Seasonality": 30,
            "Trend": 2
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

    config = dict(dataset_config[dataset_name], **train_config)

    return config
