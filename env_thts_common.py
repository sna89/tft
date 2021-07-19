from typing import List, Union
from dataclasses import dataclass, field
from torch import Tensor
from thts.node import DecisionNode
import os


@dataclass
class State:
    series: int
    steps_from_alert: int
    restart_steps: int
    temperature: float = 0
    history: List[float] = field(default_factory=list)

    def __init__(self, series, steps_from_alert, restart_steps, temperature, history):
        self.series = series
        self.steps_from_alert = steps_from_alert
        self.restart_steps = restart_steps
        self.temperature = round(temperature, 3)
        self.history = history

    def __eq__(self, other):
        if isinstance(other, State):
            if other.series == self.series and other.temperature == self.temperature:
                for idx, hist_temperature in enumerate(self.history):
                    if hist_temperature != other.history[idx]:
                        return False
                return True
            else:
                return False
        return False


@dataclass
class EnvState:
    env_state: List[State] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, EnvState):
            for idx, state in enumerate(self.env_state):
                if state != other.env_state[idx]:
                    return False
            return True
        return False


def get_reward(env_name, config, next_state_values, current_state, action: Union[List[int], int]):
    assert env_name in ["simulation", "real"]
    max_steps_from_alert = config["Env"]["AlertMaxPredictionSteps"] + 1
    min_steps_from_alert = config["Env"]["AlertMinPredictionSteps"] + 1
    max_restart_steps = config.get("Env").get("RestartSteps") + 1

    reward = 0
    any_false_alert = False
    false_alert = False
    good_alert = False
    missed_alert = False
    num_series = len(next_state_values)
    steps_from_alert = current_state.env_state[0].steps_from_alert

    if env_name == "simulation":
        action = [action] * num_series

    for series in range(num_series):
        restart_steps = current_state.env_state[series].restart_steps
        steps_from_alert = current_state.env_state[series].steps_from_alert

        if restart_steps < max_restart_steps:
            continue

        good_alert, false_alert, missed_alert = get_reward_type_for_series(config,
                                                                           series,
                                                                           next_state_values,
                                                                           current_state,
                                                                           max_steps_from_alert,
                                                                           min_steps_from_alert,
                                                                           action[series])
        if env_name == "simulation":
            if false_alert:
                any_false_alert = True
            if good_alert or missed_alert:
                break
        elif env_name == "real":
            reward += calc_reward(config,
                                  good_alert,
                                  false_alert,
                                  missed_alert,
                                  steps_from_alert,
                                  max_steps_from_alert)

    if env_name == "simulation":
        if any_false_alert and not good_alert:
            false_alert = True
        reward += calc_reward(config,
                              good_alert,
                              false_alert,
                              missed_alert,
                              steps_from_alert,
                              max_steps_from_alert)

    return reward


def get_reward_type_for_series(config,
                               num_series,
                               next_state_values,
                               current_state,
                               max_steps_from_alert,
                               min_steps_from_alert,
                               action: int):
    lb, ub = get_series_lower_and_upper_bounds(config, num_series)
    next_state_value = next_state_values[num_series]
    steps_from_alert = current_state.env_state[num_series].steps_from_alert

    good_alert = False
    false_alert = False
    missed_alert = False

    if is_missed_alert(lb, ub, next_state_value, steps_from_alert, max_steps_from_alert, action):
        missed_alert = True
    elif is_false_alert(lb, ub, next_state_value, steps_from_alert, min_steps_from_alert):
        false_alert = True
    elif is_good_alert(lb, ub, next_state_value, steps_from_alert, max_steps_from_alert, action):
        good_alert = True
    else:
        pass
    return good_alert, false_alert, missed_alert


def calc_reward(config: dict,
                good_alert: bool,
                false_alert: bool,
                missed_alert: bool,
                steps_from_alert: int,
                max_steps_from_alert: int):
    reward_false_alert = config["Env"]["Rewards"]["FalseAlert"]
    reward_missed_alert = config["Env"]["Rewards"]["MissedAlert"]
    reward_good_alert = config["Env"]["Rewards"]["GoodAlert"]

    if good_alert:
        return calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert)
    elif false_alert:
        return reward_false_alert
    elif missed_alert:
        return reward_missed_alert
    else:
        return 0


def is_missed_alert(lb, ub, next_state_value, steps_from_alert, max_steps_from_alert, action):
    if is_value_out_of_bound(next_state_value, lb, ub) and (steps_from_alert == max_steps_from_alert) \
            and not action == 1:
        return True
    return False


def is_false_alert(lb, ub, next_state_value, steps_from_alert, min_steps_from_alert):
    if not is_value_out_of_bound(next_state_value, lb, ub) and steps_from_alert == min_steps_from_alert:
        return True
    return False


def is_good_alert(lb, ub, next_state_value, steps_from_alert, max_steps_from_alert, action):
    if is_value_out_of_bound(next_state_value, lb, ub) and \
            ((steps_from_alert < max_steps_from_alert) or
             (steps_from_alert == max_steps_from_alert and action == 1)):
        return True
    return False


def calc_good_alert_reward(steps_from_alert, max_steps_from_alert, reward_good_alert):
    return reward_good_alert * (max_steps_from_alert - steps_from_alert + 1)


def build_next_state(env_name,
                     config,
                     current_state: EnvState,
                     next_state_values: List,
                     max_steps_from_alert: int,
                     max_restart_steps: int,
                     action: Union[List[int], int]):
    assert env_name in ["simulation", "real"], "{} is not supported".format(env_name)

    next_state = EnvState()
    group_names = list(config.get("AnomalyConfig").get(os.getenv("DATASET")).keys())
    terminal_states = []

    if env_name == "simulation" and isinstance(action, int):
        action = [action] * len(group_names)

    for idx, group_name in enumerate(group_names):
        group_state = get_group_state(current_state.env_state, group_name)

        steps_from_alert = update_steps_from_alert(group_state.steps_from_alert,
                                                   max_steps_from_alert,
                                                   action[idx])

        lb, ub = get_series_lower_and_upper_bounds(config, group_name)
        out_of_bound = is_value_out_of_bound(next_state_values[idx], lb, ub)
        terminal_states.append(out_of_bound)
        restart_steps = update_restart_steps(out_of_bound,
                                             group_state.restart_steps,
                                             max_restart_steps)

        next_state_series_history = group_state.history + \
                                    [group_state.temperature]

        if isinstance(next_state_values[idx], Tensor):
            next_state_series_temperature = next_state_values[idx].item()
        else:
            next_state_series_temperature = next_state_values[idx]

        next_state_series = State(group_name,
                                  steps_from_alert,
                                  restart_steps,
                                  next_state_series_temperature,
                                  next_state_series_history)
        next_state.env_state.append(next_state_series)
    return next_state, terminal_states


def update_steps_from_alert(steps_from_alert, max_steps_from_alert, action):
    if action == 1 or (action == 0 and steps_from_alert < max_steps_from_alert):
        steps_from_alert -= 1
        if steps_from_alert == 0:
            steps_from_alert = max_steps_from_alert
    return steps_from_alert


def update_restart_steps(out_of_bounds: bool, restart_steps: int, max_restart_steps: int):
    if restart_steps < max_restart_steps or out_of_bounds:
        restart_steps -= 1
        if restart_steps == 0:
            restart_steps = max_restart_steps
    return restart_steps


def is_alertable_state(current_node: DecisionNode, max_alert_prediction_steps: int, max_restart_env_iterations: int):
    if all(series_state.steps_from_alert < max_alert_prediction_steps or
           series_state.restart_steps < max_restart_env_iterations
           for series_state in current_node.state.env_state):
        return False
    return True


def get_series_lower_and_upper_bounds(config, group):
    bounds = config.get("AnomalyConfig").get(os.getenv("DATASET")).get(group)
    lb, ub = bounds.values()
    return lb, ub


def is_value_out_of_bound(value, lb, ub):
    if value < lb or value > ub:
        return True
    return False


def get_group_state(env_state, group):
    for group_state in env_state:
        if group == group_state.series:
            return group_state