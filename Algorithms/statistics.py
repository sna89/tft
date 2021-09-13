

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


def update_statistics(config, group_names, statistics, reward_group_mapping, action_group_mapping_dict):
    reward_false_alert = config.get("Env").get("Rewards").get("FalseAlert")
    reward_missed_alert = config.get("Env").get("Rewards").get("MissedAlert")

    for group_name in group_names:
        group_statistics = statistics[group_name]
        reward = reward_group_mapping[group_name]
        action = action_group_mapping_dict[group_name]

        if reward == 0:
            group_statistics["TN"] += 1
        elif reward > 0:
            group_statistics["TP"] += 1
        elif reward < 0:
            if reward_missed_alert != reward_false_alert:
                if reward == reward_missed_alert:
                    group_statistics["FN"] += 1
                elif reward == reward_false_alert:
                    group_statistics["FP"] += 1
            else:
                if action == 0:
                    group_statistics["FN"] += 1
                elif action == 1:
                    group_statistics["FP"] += 1
    return statistics