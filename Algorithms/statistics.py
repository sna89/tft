from env_thts_common import get_steps_from_alert


def init_statistics(group_names):
    statistics = {}
    for group_name in group_names:
        statistics[group_name] = {
            "TP": 0,
            "FN": 0,
            "FP": 0,
            "TN": 0
        }
    return statistics


def update_statistics(config,
                      group_names,
                      statistics,
                      group_reward_mapping,
                      group_name_steps_from_alert_mapping):
    steps_from_alert = get_steps_from_alert(config)

    for group_name in group_names:
        group_statistics = statistics[group_name]
        reward = group_reward_mapping[group_name]
        group_steps_from_alert = group_name_steps_from_alert_mapping[group_name]

        if reward == 0:
            group_statistics["TN"] += 1
        elif reward > 0:
            group_statistics["TP"] += 1
        elif reward < 0:
            if group_steps_from_alert == steps_from_alert:
                group_statistics["FN"] += 1
            elif group_steps_from_alert == 1:
                group_statistics["FP"] += 1
    return statistics
