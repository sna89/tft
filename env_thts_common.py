import os
from typing import List, Dict, Union
from dataclasses import dataclass, field
from torch import Tensor
from Algorithms.thts.node import DecisionNode
from data_utils import get_group_lower_and_upper_bounds
from config import DATETIME_COLUMN
import pandas as pd


@dataclass
class State:
    group: int
    steps_from_alert: int
    restart_steps: int
    value: float = 0
    history: List[float] = field(default_factory=list)

    def __init__(self, group, steps_from_alert, restart_steps, value, history):
        self.group = group
        self.steps_from_alert = steps_from_alert
        self.restart_steps = restart_steps
        self.value = round(value, 3)
        self.history = history

    def __eq__(self, other):
        if isinstance(other, State):
            if other.group == self.group and other.value == self.value:
                for idx, hist_value in enumerate(self.history):
                    if hist_value != other.history[idx]:
                        return False
                return True
            else:
                return False
        return False


@dataclass
class EnvState:
    env_state: Dict[str, State] = field(default_factory=dict)

    # def __eq__(self, other):
    #     if isinstance(other, EnvState):
    #         for idx, state in enumerate(self.env_state):
    #             if state != other.env_state[idx]:
    #                 return False
    #         return True
    #     return False


def build_group_next_state(config,
                           group_next_state_value,
                           group_current_state,
                           restart_steps,
                           steps_from_alert,
                           action,
                           group_name
                           ):
    is_group_current_state_restart = is_state_restart(group_current_state.restart_steps,
                                                      restart_steps)
    is_group_current_state_terminal = is_state_terminal(config,
                                                        group_name,
                                                        group_current_state.value,
                                                        is_group_current_state_restart)

    group_next_steps_from_alert = update_steps_from_alert(is_group_current_state_restart,
                                                          is_group_current_state_terminal,
                                                          group_current_state.steps_from_alert,
                                                          steps_from_alert,
                                                          action)
    is_group_next_state_terminal = False \
        if is_group_current_state_restart and group_current_state.restart_steps > 1 \
        else is_state_terminal(config, group_name, group_next_state_value)

    group_next_state_restart_steps = update_restart_steps(is_group_current_state_terminal,
                                                          group_current_state.restart_steps,
                                                          restart_steps)
    is_group_next_state_restart = False \
        if is_group_next_state_terminal \
        else is_state_restart(group_next_state_restart_steps, restart_steps)

    group_next_state_history = group_current_state.history + [group_current_state.value]
    group_next_state_value = group_next_state_value.item() if isinstance(group_next_state_value,
                                                                         Tensor) else group_next_state_value

    group_next_state = State(group_name,
                             group_next_steps_from_alert,
                             group_next_state_restart_steps,
                             group_next_state_value,
                             group_next_state_history)

    return group_next_state, is_group_next_state_terminal, is_group_next_state_restart


def build_next_state(env_name,
                     config,
                     current_state: EnvState,
                     group_names,
                     next_state_values: Dict,
                     steps_from_alerts: int,
                     restart_steps: int,
                     action_dict: Dict):
    assert env_name in ["simulation", "real"], "{} is not supported".format(env_name)

    next_state = EnvState()
    terminal_states = {}
    restart_states = {}

    for group_name in group_names:
        action = action_dict[group_name]
        group_current_state = get_group_state(current_state.env_state, str(group_name))
        group_next_state_value = next_state_values[group_name]

        group_next_state, is_group_terminal, is_group_restart = build_group_next_state(config,
                                                                                       group_next_state_value,
                                                                                       group_current_state,
                                                                                       restart_steps,
                                                                                       steps_from_alerts,
                                                                                       action,
                                                                                       group_name)
        terminal_states[group_name] = is_group_terminal
        restart_states[group_name] = is_group_restart
        next_state.env_state[group_name] = group_next_state

    return next_state, terminal_states, restart_states


def get_reward(env_name, config, group_names, next_state_terminal_dict, current_state, action_dict: Dict) \
        -> Union[Dict, float]:
    assert env_name in ["simulation", "real"]
    env_steps_from_alert = get_env_steps_from_alert(config)
    env_restart_steps = get_env_restart_steps(config)

    reward_group_mapping = {}

    for group_name in group_names:
        group_current_state = get_group_state(current_state.env_state, group_name)

        is_group_current_state_restart = is_state_restart(group_current_state.restart_steps,
                                                          env_restart_steps)
        is_group_current_state_terminal = is_state_terminal(config,
                                                            group_name,
                                                            group_current_state.value,
                                                            is_group_current_state_restart)
        if is_group_current_state_restart or is_group_current_state_terminal:
            reward_group_mapping[group_name] = 0
            continue

        good_alert, false_alert, missed_alert = get_reward_type_for_group(group_current_state.steps_from_alert,
                                                                          env_steps_from_alert,
                                                                          next_state_terminal_dict[group_name],
                                                                          action_dict[group_name])

        reward = calc_reward(config,
                             good_alert,
                             false_alert,
                             missed_alert,
                             group_current_state.steps_from_alert,
                             env_steps_from_alert)
        reward_group_mapping[group_name] = reward

    return reward_group_mapping


def get_reward_type_for_group(group_current_steps_from_alert: int,
                              env_steps_from_alert: int,
                              is_group_next_state_terminal: bool,
                              group_action: int):
    good_alert = False
    false_alert = False
    missed_alert = False

    if is_missed_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                       group_action):
        missed_alert = True
    elif is_false_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                        group_action):
        false_alert = True
    elif is_good_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                       group_action):
        good_alert = True
    else:
        pass
    return good_alert, false_alert, missed_alert


def calc_reward(config: dict,
                good_alert: bool,
                false_alert: bool,
                missed_alert: bool,
                current_steps_from_alert: int,
                steps_from_alert: int):
    assert os.getenv("REWARD_TYPE") in ["CheapFP", "ExpensiveFP"]
    reward_false_alert = config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("FalseAlert")
    reward_missed_alert = config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("MissedAlert")
    reward_good_alert = config.get("Env").get("Rewards").get(os.getenv("REWARD_TYPE")).get("GoodAlert")

    if good_alert:
        return calc_good_alert_reward(current_steps_from_alert, steps_from_alert, reward_good_alert)
    elif false_alert:
        return reward_false_alert
    elif missed_alert:
        return reward_missed_alert
    else:
        return 0


def is_missed_alert(next_state_terminal, current_steps_from_alert, env_steps_from_alert, action):
    if next_state_terminal and \
            action == 0 and \
            (current_steps_from_alert == env_steps_from_alert or env_steps_from_alert == 1):
        return True
    return False


def is_false_alert(is_next_state_terminal, current_steps_from_alert, env_steps_from_alert, action):
    if env_steps_from_alert == 1:
        if action == 1 and not is_next_state_terminal:
            return True
    elif env_steps_from_alert > 1:
        if action == 0 and current_steps_from_alert == 1 and not is_next_state_terminal:
            return True
    return False


def is_good_alert(next_state_terminal, current_steps_from_alert, steps_from_alert, action):
    if next_state_terminal and ((current_steps_from_alert < steps_from_alert and action == 0) or
                                (current_steps_from_alert == steps_from_alert and action == 1)):
        return True
    return False


def calc_good_alert_reward(current_steps_from_alert, max_steps_from_alert, reward_good_alert):
    return reward_good_alert * (max_steps_from_alert - (current_steps_from_alert - 1))


def is_alertable_state(current_node: DecisionNode, max_alert_prediction_steps: int, max_restart_env_iterations: int):
    if all(series_state.steps_from_alert < max_alert_prediction_steps or
           series_state.restart_steps < max_restart_env_iterations
           for _, series_state in current_node.state.env_state.items()):
        return False
    return True


def get_group_state(env_state, group_name):
    return env_state[group_name]


def update_steps_from_alert(current_state_restart,
                            current_state_terminal,
                            current_steps_from_alert,
                            env_steps_from_alert,
                            action):
    next_steps_from_alert = current_steps_from_alert
    if env_steps_from_alert > 1:
        if current_state_terminal:
            next_steps_from_alert = env_steps_from_alert

        else:
            if not current_state_terminal and not current_state_restart and \
                    (action == 1 or (action == 0 and current_steps_from_alert < env_steps_from_alert)):
                next_steps_from_alert = current_steps_from_alert - 1

                if current_steps_from_alert == 0:
                    next_steps_from_alert = env_steps_from_alert

    return next_steps_from_alert


def is_state_terminal(config, group_name, value, current_state_restart=None):
    if not current_state_restart:
        lb, ub = get_group_lower_and_upper_bounds(config, group_name)
        if value < lb or value > ub:
            return True
    return False


def update_restart_steps(current_state_terminal: bool, current_restart_steps: int, env_restart_steps: int):
    next_restart_steps = current_restart_steps
    if current_state_terminal:
        next_restart_steps = env_restart_steps - 1 if env_restart_steps > 1 else env_restart_steps

    else:
        if is_state_restart(current_restart_steps, env_restart_steps):
            if current_restart_steps == 0:
                next_restart_steps = env_restart_steps
            else:
                next_restart_steps = current_restart_steps - 1

    return next_restart_steps


def get_group_names(group_name_group_idx_mapping):
    return list(group_name_group_idx_mapping.values())


def is_state_restart(restart_steps, env_restart_steps):
    return restart_steps < env_restart_steps


def get_num_iterations(config, test_df):
    enc_len = config.get("EncoderLength")
    pred_len = config.get("PredictionLength")
    num_iterations = test_df.time_idx.max() - test_df.time_idx.min() - enc_len - pred_len
    return num_iterations


def get_last_val_time_idx(config, test_df):
    return test_df.time_idx.min() + config.get("EncoderLength") + config.get("PredictionLength") - 2


def get_last_val_date(test_df, last_val_time_idx):
    if DATETIME_COLUMN in test_df.columns:
        last_date = test_df[test_df.time_idx == last_val_time_idx][DATETIME_COLUMN].unique()[0]
        return pd.to_datetime(last_date)
    return None


def get_env_steps_from_alert(config):
    return config.get("Env").get("AlertMaxPredictionSteps")


def get_restart_steps(config):
    return config.get("Env").get("RestartSteps")


def get_env_restart_steps(config):
    return get_restart_steps(config) + 1
