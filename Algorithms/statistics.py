import os
from EnvCommon.env_thts_common import get_env_steps_from_alert
from config import get_false_alert_reward, get_missed_alert_reward, get_task_folder_name
from torch import Tensor
import pandas as pd
from config import REGRESSION_TASK_TYPE, RESULT
from evaluation import evaluate_classification_from_conf_matrix


def init_statistics(group_names):
    statistics = {}
    if isinstance(group_names, (list, Tensor)):
        for group_name in group_names:
            if isinstance(group_name, Tensor):
                group_name = group_name.item()
            statistics[group_name] = {
                "TP": 0,
                "FN": 0,
                "FP": 0,
                "TN": 0
            }
    elif isinstance(group_names, str):
        statistics[group_names] = {
            "TP": 0,
            "FN": 0,
            "FP": 0,
            "TN": 0
        }
    else:
        raise ValueError
    return statistics


def update_statistics(config,
                      group_name,
                      statistics,
                      reward,
                      group_steps_from_alert):
    env_steps_from_alert = get_env_steps_from_alert(config)
    false_positive_reward = get_false_alert_reward(config)
    false_negative_reward = get_missed_alert_reward(config)
    group_statistics = statistics[group_name]

    if reward == 0:
        group_statistics["TN"] += 1
    elif reward > 0:
        group_statistics["TP"] += 1
    elif reward < 0:
        if group_steps_from_alert == env_steps_from_alert and \
                reward == false_negative_reward:
            group_statistics["FN"] += 1

        elif group_steps_from_alert == 1 and reward == false_positive_reward:
            group_statistics["FP"] += 1
    return statistics


def output_statistics(statistics, reward, filter_group_name=None, task_type=REGRESSION_TASK_TYPE):
    for group_name, group_statistics in statistics.items():
        if not filter_group_name or group_name == filter_group_name:
            evaluation_dict = dict()
            if isinstance(reward, int):
                evaluation_dict["Cumulative Reward"] = [reward]
            elif isinstance(reward, dict):
                evaluation_dict["Cumulative Reward"] = [reward[group_name]]

            tp, fn, fp, tn = list(group_statistics.values())
            precision, recall, f_score = evaluate_classification_from_conf_matrix(tn, fp, fn, tp)
            print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))
            evaluation_dict["TN"] = [tn]
            evaluation_dict["FP"] = [fp]
            evaluation_dict["FN"] = [fn]
            evaluation_dict["TP"] = [tp]
            evaluation_dict["Precision"] = [precision]
            evaluation_dict["Recall"] = [recall]
            evaluation_dict["F_score"] = [f_score]
            evaluation_df = pd.DataFrame.from_dict(evaluation_dict)

            folder_name = get_task_folder_name(task_type)

            evaluation_df.to_csv(os.path.join(RESULT,
                                              folder_name,
                                              "evaluation_{}.csv".format(group_name)))



