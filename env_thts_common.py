from typing import List, Dict, Union
from dataclasses import dataclass, field
from torch import Tensor
from Algorithms.thts.node import DecisionNode
import os
import torch


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


def build_group_next_state(config,
                           next_state_value,
                           current_group_state,
                           max_restart_steps,
                           max_steps_from_alert,
                           action,
                           group_name
                           ):
    current_state_restart = is_state_restart(current_group_state.restart_steps,
                                             max_restart_steps)
    current_state_terminal = is_state_terminal(config,
                                               group_name,
                                               current_group_state.temperature,
                                               current_state_restart)

    next_steps_from_alert = update_steps_from_alert(current_state_restart,
                                                    current_state_terminal,
                                                    current_group_state.steps_from_alert,
                                                    max_steps_from_alert,
                                                    action)
    next_state_terminal = False \
        if current_state_restart and current_group_state.restart_steps > 1 \
        else is_state_terminal(config, group_name, next_state_value)
    next_state_restart_steps = update_restart_steps(current_state_terminal,
                                                    current_group_state.restart_steps,
                                                    max_restart_steps)
    next_state_restart = False \
        if next_state_terminal \
        else is_state_restart(next_state_restart_steps, max_restart_steps)

    next_state_history = current_group_state.history + [current_group_state.temperature]
    next_state_temperature = next_state_value.item() if isinstance(next_state_value, Tensor) else next_state_value

    group_next_state = State(group_name,
                             next_steps_from_alert,
                             next_state_restart_steps,
                             next_state_temperature,
                             next_state_history)

    return group_next_state, next_state_terminal, next_state_restart


def build_next_state(env_name,
                     config,
                     current_state: EnvState,
                     group_names,
                     next_state_values: Dict,
                     max_steps_from_alert: int,
                     max_restart_steps: int,
                     action_dict: Dict):
    assert env_name in ["simulation", "real"], "{} is not supported".format(env_name)

    next_state = EnvState()
    terminal_states = {}
    restart_states = {}

    for group_name in group_names:
        action = action_dict[group_name]
        current_group_state = get_group_state(current_state.env_state, str(group_name))
        next_state_value = next_state_values[group_name]

        group_next_state, terminal, restart = build_group_next_state(config,
                                                                     next_state_value,
                                                                     current_group_state,
                                                                     max_restart_steps,
                                                                     max_steps_from_alert,
                                                                     action,
                                                                     group_name)
        terminal_states[group_name] = terminal
        restart_states[group_name] = restart
        next_state.env_state.append(group_next_state)

    return next_state, terminal_states, restart_states


def get_reward(env_name, config, group_names, next_state_terminal_dict, current_state, action_dict: Dict) \
        -> Union[Dict, float]:
    assert env_name in ["simulation", "real"]
    max_steps_from_alert = config.get("Env").get("AlertMaxPredictionSteps") + 1
    min_steps_from_alert = config.get("Env").get("AlertMinPredictionSteps") + 1
    max_restart_steps = config.get("Env").get("RestartSteps") + 1

    reward_group_mapping = {}

    for group_name in group_names:
        reward = 0
        current_group_state = get_group_state(current_state.env_state, group_name)

        current_state_restart = is_state_restart(current_group_state.restart_steps,
                                                 max_restart_steps)
        current_state_terminal = is_state_terminal(config,
                                                   group_name,
                                                   current_group_state.temperature,
                                                   current_state_restart)
        if current_state_restart or current_state_terminal:
            reward_group_mapping[group_name] = reward
            continue

        next_state_terminal = next_state_terminal_dict[group_name]
        action = action_dict[group_name]

        good_alert, false_alert, missed_alert = get_reward_type_for_group(current_group_state.steps_from_alert,
                                                                          min_steps_from_alert,
                                                                          max_steps_from_alert,
                                                                          next_state_terminal,
                                                                          action)

        reward += calc_reward(config,
                              good_alert,
                              false_alert,
                              missed_alert,
                              current_group_state.steps_from_alert,
                              max_steps_from_alert)
        reward_group_mapping[group_name] = reward

    if env_name == "simulation":
        reward = sum([reward for group_name, reward in reward_group_mapping.items()]) / float(len(reward_group_mapping))
    elif env_name == "real":
        reward = reward_group_mapping
    else:
        raise ValueError
    return reward


def get_reward_type_for_group(current_steps_from_alert: int,
                              min_steps_from_alert: int,
                              max_steps_from_alert: int,
                              next_state_terminal: bool,
                              action: int):
    good_alert = False
    false_alert = False
    missed_alert = False

    if is_missed_alert(next_state_terminal, current_steps_from_alert, max_steps_from_alert, action):
        missed_alert = True
    elif is_false_alert(next_state_terminal, current_steps_from_alert, min_steps_from_alert):
        false_alert = True
    elif is_good_alert(next_state_terminal, current_steps_from_alert, max_steps_from_alert, action):
        good_alert = True
    else:
        pass
    return good_alert, false_alert, missed_alert


def calc_reward(config: dict,
                good_alert: bool,
                false_alert: bool,
                missed_alert: bool,
                current_steps_from_alert: int,
                max_steps_from_alert: int):

    reward_false_alert = config.get("Env").get("Rewards").get("FalseAlert")
    reward_missed_alert = config.get("Env").get("Rewards").get("MissedAlert")
    reward_good_alert = config.get("Env").get("Rewards").get("GoodAlert")

    if good_alert:
        return calc_good_alert_reward(current_steps_from_alert, max_steps_from_alert, reward_good_alert)
    elif false_alert:
        return reward_false_alert
    elif missed_alert:
        return reward_missed_alert
    else:
        return 0


def is_missed_alert(next_state_terminal, current_steps_from_alert, max_steps_from_alert, action):
    if next_state_terminal and current_steps_from_alert == max_steps_from_alert and action == 0:
        return True
    return False


def is_false_alert(next_state_terminal, current_steps_from_alert, min_steps_from_alert):
    if not next_state_terminal and current_steps_from_alert == min_steps_from_alert:
        return True
    return False


def is_good_alert(next_state_terminal, current_steps_from_alert, max_steps_from_alert, action):
    if next_state_terminal and ((current_steps_from_alert < max_steps_from_alert) or
                                (current_steps_from_alert == max_steps_from_alert and action == 1)):
        return True
    return False


def calc_good_alert_reward(current_steps_from_alert, max_steps_from_alert, reward_good_alert):
    return reward_good_alert * (max_steps_from_alert - (current_steps_from_alert - 1))


def is_alertable_state(current_node: DecisionNode, max_alert_prediction_steps: int, max_restart_env_iterations: int):
    if all(series_state.steps_from_alert < max_alert_prediction_steps or
           series_state.restart_steps < max_restart_env_iterations
           for series_state in current_node.state.env_state):
        return False
    return True


def get_group_state(env_state, group):
    for group_state in env_state:
        if group == group_state.series:
            return group_state


def update_steps_from_alert(current_state_restart,
                            current_state_terminal,
                            steps_from_alert,
                            max_steps_from_alert,
                            action):
    if not current_state_terminal and not current_state_restart and \
            (action == 1 or (action == 0 and steps_from_alert < max_steps_from_alert)):
        steps_from_alert -= 1
        if steps_from_alert == 0:
            steps_from_alert = max_steps_from_alert
    return steps_from_alert


def get_group_lower_and_upper_bounds(config, group_name):
    bounds = config.get("AnomalyConfig").get(os.getenv("DATASET")).get(group_name)
    lb, ub = bounds.values()
    return lb, ub


def is_state_terminal(config, group_name, value, current_state_restart=None):
    if not current_state_restart:
        lb, ub = get_group_lower_and_upper_bounds(config, group_name)
        if value < lb or value > ub:
            return True
    return False


def update_restart_steps(current_state_terminal: bool, current_restart_steps: int, max_restart_steps: int):
    if current_state_terminal:
        return max_restart_steps - 1

    else:
        if is_state_restart(current_restart_steps, max_restart_steps):
            current_restart_steps -= 1
            if current_restart_steps == 0:
                current_restart_steps = max_restart_steps

    return current_restart_steps


def get_group_names(group_idx_mapping):
    return list(group_idx_mapping.keys())


def is_state_restart(restart_steps, max_restart_steps):
    return restart_steps < max_restart_steps


def get_num_iterations(test_df, enc_len):
    num_iterations = test_df.time_idx.max() - test_df.time_idx.min() - enc_len + 3
    return num_iterations


def get_group_idx_mapping(config, model, test_df):
    if isinstance(model.hparams.embedding_labels, dict) and \
            config.get("GroupKeyword") in model.hparams.embedding_labels:
        return model.hparams.embedding_labels[config.get("GroupKeyword")]
    else:
        group_name_list = list(test_df[config.get("GroupKeyword")].unique())
        return {group_name: group_name for group_name in group_name_list}


def is_group_prediction_out_of_bound(group_prediction, lb, ub):
    out_of_bound = torch.where((group_prediction < lb) | (group_prediction > ub), 1, 0)
    if sum(out_of_bound) > 0:
        idx = (out_of_bound == 1).nonzero()[0].item()
        return True, idx
    else:
        return False, -1
