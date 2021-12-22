import os
from config import REGRESSION_TASK_TYPE, CLASSIFICATION_TASK_TYPE, COMBINED_TASK_TYPE


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