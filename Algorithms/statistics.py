from EnvCommon.env_thts_common import get_env_steps_from_alert


def init_statistics(group_names):
    statistics = {}
    if isinstance(group_names, list):
        for group_name in group_names:
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
    group_statistics = statistics[group_name]

    if reward == 0:
        group_statistics["TN"] += 1
    elif reward > 0:
        group_statistics["TP"] += 1
    elif reward < 0:
        if group_steps_from_alert == env_steps_from_alert:
            group_statistics["FN"] += 1
        elif group_steps_from_alert == 1:
            group_statistics["FP"] += 1
    return statistics
