import os


class DataConst:
    DATASET_NAME = "Electricity"
    PREDICTION_LENGTH = 24
    ENCODER_LENGTH = 100


class HyperParameters:
    BATCH_SIZE = 128


class DataSetRatio:
    TRAIN = 0.6
    VAL = 0.2


class Paths:
    # BASE_FOLDER = os.path.join('/', 'tmp', 'pycharm_project_269', 'Data')
    BASE_FOLDER = os.path.join('Data')
    FISHERMAN = os.path.join(BASE_FOLDER, 'Fisherman')
    ELECTRICITY = os.path.join(BASE_FOLDER, 'Electricity')
