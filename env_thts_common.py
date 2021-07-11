

def get_reward_and_terminal(config, prediction, steps_from_alert):
    reward_false_alert = config["Env"]["Rewards"]["FalseAlert"]
    reward_missed_alert = config["Env"]["Rewards"]["MissedAlert"]
    reward_good_alert = config["Env"]["Rewards"]["GoodAlert"]

    alert_prediction_steps = config["Env"]["AlertMaxPredictionSteps"]
    min_steps_from_alert = config["Env"]["AlertMinPredictionSteps"]
    max_steps_from_alert = alert_prediction_steps + 1

    reward = 0
    terminal = False
    for num_series in range(len(prediction)):
        bounds = config.get("AnomalyConfig").get("series_{}".format(num_series))
        lb, hb = bounds.values()
        series_prediction = prediction[num_series]
        if is_missed_alert(lb, hb, series_prediction, steps_from_alert, max_steps_from_alert):
            reward += reward_missed_alert
            terminal = True
        if is_false_alert(lb, hb, series_prediction, steps_from_alert, min_steps_from_alert):
            reward += reward_false_alert
        if is_good_alert(lb, hb, series_prediction, steps_from_alert, max_steps_from_alert):
            reward += calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert)
            terminal = True
    return reward, terminal


def is_missed_alert(lb, hb, prediction, steps_from_alert, max_steps_from_alert):
    if (prediction < lb or prediction > hb) and (steps_from_alert == max_steps_from_alert):
        return True
    return False


def is_false_alert(lb, hb, prediction, steps_from_alert, min_steps_from_alert):
    if (lb <= prediction <= hb) and (steps_from_alert == min_steps_from_alert):
        return True
    return False


def is_good_alert(lb, hb, prediction, steps_from_alert, max_steps_from_alert):
    if (prediction < lb or prediction > hb) and (steps_from_alert < max_steps_from_alert):
        return True
    return False


def calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert):
    return reward_good_alert * (max_steps_from_alert - steps_from_alert)