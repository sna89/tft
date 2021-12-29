import pickle
from typing import List
import random
import os
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.models.deepar import DeepAR
from Models.fc import FullyConnectedModel


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


def create_chunks(l, chunk_size):
    n = len(l)
    for i in range(0, n, chunk_size):
        yield l[i: i + chunk_size]


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


def get_prediction_mode():
    model_name = os.getenv("MODEL_NAME_REG")
    if model_name == "TFT":
        mode = "raw"
    elif model_name == "DeepAR":
        mode = "quantiles"
    else:
        raise ValueError
    return mode


def flatten_nested_list(nl):
    flat_list = []
    for sublist in nl:
        for item in sublist:
            flat_list.append(item)
    return flat_list


