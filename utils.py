import pickle
from typing import List
import random
import os
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.models.deepar import DeepAR
from Models.fc import FullyConnectedModel
from config import REGRESSION_TASK_TYPE, COMBINED_TASK_TYPE, CLASSIFICATION_TASK_TYPE


def save_to_pickle(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def get_argmax_from_list(l: List, choose_random=True):
    max_value = max(l)
    max_indices = [i for i, value in enumerate(l) if value == max_value]
    if choose_random:
        max_idx = random.choice(max_indices)
        return max_idx
    return max_indices


def set_env_to_state(env, state):
    env._current_state = state


def get_model_from_trainer(trainer, model_name):
    best_model_path = trainer.checkpoint_callback.best_model_path
    os.environ["CHECKPOINT"] = best_model_path
    best_tft = get_model_from_checkpoint(best_model_path, model_name)
    return best_tft


def get_model_from_checkpoint(checkpoint, model_name):
    best_model = None
    if model_name == "TFT":
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    elif model_name == "DeepAR":
        best_model = DeepAR.load_from_checkpoint(checkpoint)
    elif model_name == "Mlp":
        best_model = FullyConnectedModel.load_from_checkpoint(checkpoint)
    return best_model


def get_study_path(config, task_type):
    if task_type == REGRESSION_TASK_TYPE:
        study_pkl_path = os.path.join(config.get("StudyRegPath"))
    elif task_type == CLASSIFICATION_TASK_TYPE:
        study_pkl_path = os.path.join(config.get("StudyClassPath"))
    elif task_type == COMBINED_TASK_TYPE:
        study_pkl_path = os.path.join(config.get("StudyCombinedPath"))
    else:
        raise ValueError
    return study_pkl_path


def get_study_pkl_path(config, task_type):
    return os.path.join(get_study_path(config, task_type), "study.pkl")
