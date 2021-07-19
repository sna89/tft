import pickle
from typing import List
import random


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
    env.current_state = state