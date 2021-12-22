from Models.fc import FullyConnectedModel
from config import REGRESSION_TASK_TYPE
from Models.trainer import get_study_pkl_path
from utils import load_pickle


def create_mlp_model(config, loss):
    study_pkl_path = get_study_pkl_path(config, REGRESSION_TASK_TYPE)
    study = load_pickle(study_pkl_path)

    input_size = study.best_params['hidden_size'] if study and 'hidden_size' in study.best_params else 64

    fc = FullyConnectedModel(input_size=input_size,
                             hidden_size=input_size * 32,
                             output_size=1,
                             n_hidden_layers=2,
                             n_classes=2,
                             dropout=0.5,
                             loss=loss,
                             learning_rate=1e-7)
    return fc