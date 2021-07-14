from typing import List
from dataclasses import dataclass, field
from torch import Tensor


@dataclass
class State:
    series: int
    temperature: float = 0
    history: List[float] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, State):
            if other.series == self.series and round(other.temperature, 3) == round(self.temperature, 3):
                for idx, hist_temperature in enumerate(self.history):
                    if round(hist_temperature, 3) != round(other.history[idx], 3):
                        return False
                return True
            else:
                return False
        return False


@dataclass
class EnvState:
    env_state: List[State] = field(default_factory=list)
    steps_from_alert: int = -1

    def __eq__(self, other):
        if isinstance(other, EnvState):
            for idx, state in enumerate(self.env_state):
                if state != other.env_state[idx]:
                    return False
            if self.steps_from_alert != other.steps_from_alert:
                return False
            return True
        return False


def get_reward_and_terminal(config, next_state_values, steps_from_alert, action):
    reward_false_alert = config["Env"]["Rewards"]["FalseAlert"]
    reward_missed_alert = config["Env"]["Rewards"]["MissedAlert"]
    reward_good_alert = config["Env"]["Rewards"]["GoodAlert"]

    max_steps_from_alert = config["Env"]["AlertMaxPredictionSteps"] + 1
    min_steps_from_alert = config["Env"]["AlertMinPredictionSteps"] + 1

    reward = 0
    terminal = False
    for num_series in range(len(next_state_values)):
        bounds = config.get("AnomalyConfig").get("series_{}".format(num_series))
        lb, hb = bounds.values()
        series_prediction = next_state_values[num_series]
        if is_missed_alert(lb, hb, series_prediction, steps_from_alert, max_steps_from_alert, action):
            reward += reward_missed_alert
            terminal = True
            break
        if is_false_alert(lb, hb, series_prediction, steps_from_alert, min_steps_from_alert):
            reward += reward_false_alert
            break
        if is_good_alert(lb, hb, series_prediction, steps_from_alert, max_steps_from_alert, action):
            reward += calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert)
            terminal = True
            break
    return reward, terminal


def is_missed_alert(lb, hb, prediction, steps_from_alert, max_steps_from_alert, action):
    if (prediction < lb or prediction > hb) and (steps_from_alert == max_steps_from_alert) \
            and not action:
        return True
    return False


def is_false_alert(lb, hb, prediction, steps_from_alert, min_steps_from_alert):
    if (lb <= prediction <= hb) and (steps_from_alert == min_steps_from_alert):
        return True
    return False


def is_good_alert(lb, hb, prediction, steps_from_alert, max_steps_from_alert, action):
    if (prediction < lb or prediction > hb) and \
            ((steps_from_alert < max_steps_from_alert) or
             (steps_from_alert == max_steps_from_alert and action == 1)):
        return True
    return False


def calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert):
    return reward_good_alert * (max_steps_from_alert - steps_from_alert)


def build_next_state(current_state: EnvState, next_state_values: List, max_steps_from_alert: int, action: int):
    next_state = EnvState()

    next_state.steps_from_alert = update_steps_from_alert(current_state.steps_from_alert,
                                                          max_steps_from_alert,
                                                          action)

    for series, value in enumerate(range(len(next_state_values))):
        series_state = current_state.env_state[series]
        next_state_series_history = series_state.history + [series_state.temperature]
        if isinstance(next_state_values[series], Tensor):
            next_state_series_temperature = next_state_values[series].item()
        else:
            next_state_series_temperature = next_state_values[series]
        next_state_series = State(series,
                                  next_state_series_temperature,
                                  next_state_series_history)
        next_state.env_state.append(next_state_series)
    return next_state


def update_steps_from_alert(steps_from_alert, max_steps_from_alert, action):
    if action == 1 or (action == 0 and steps_from_alert < max_steps_from_alert):
        steps_from_alert -= 1
        if steps_from_alert == 0:
            steps_from_alert = max_steps_from_alert
    return steps_from_alert
