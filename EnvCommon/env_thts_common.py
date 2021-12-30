import os
from typing import List, Dict, Union
from dataclasses import dataclass, field
from torch import Tensor
from Algorithms.thts.node import DecisionNode
from data_utils import get_group_lower_and_upper_bounds
from config import DATETIME_COLUMN, get_env_steps_from_alert, get_env_restart_steps, get_false_alert_reward, \
    get_missed_alert_reward, get_good_alert_reward, get_num_quantiles
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
                           env_restart_steps,
                           env_steps_from_alert,
                           action,
                           group_name,
                           is_env_group=False):
    if is_env_group:
        is_group_current_state_restart = is_state_restart(group_current_state.restart_steps,
                                                          env_restart_steps)
        is_group_current_state_terminal = is_state_terminal(config,
                                                            group_name,
                                                            group_current_state.value,
                                                            is_group_current_state_restart)

        group_next_steps_from_alert = update_steps_from_alert(is_group_current_state_restart,
                                                              is_group_current_state_terminal,
                                                              group_current_state.steps_from_alert,
                                                              env_steps_from_alert,
                                                              action)
        is_group_next_state_terminal = False \
            if is_group_current_state_restart and group_current_state.restart_steps > 1 \
            else is_state_terminal(config, group_name, group_next_state_value)

        group_next_state_restart_steps = update_restart_steps(is_group_next_state_terminal,
                                                              group_current_state.restart_steps,
                                                              env_restart_steps)
        is_group_next_state_restart = False \
            if is_group_next_state_terminal \
            else is_state_restart(group_next_state_restart_steps, env_restart_steps)
    else:
        group_next_steps_from_alert = group_current_state.steps_from_alert
        group_next_state_restart_steps = group_current_state.restart_steps
        is_group_next_state_terminal = False
        is_group_next_state_restart = False

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
                     current_group_name: str,
                     group_names: Dict,
                     next_state_values: Dict,
                     env_steps_from_alerts: int,
                     env_restart_steps: int,
                     action: int):
    assert env_name in ["simulation", "real"], "{} is not supported".format(env_name)

    next_state = EnvState()
    is_group_next_terminal = False
    is_group_next_restart = False

    for group_name in group_names:
        is_current_group = group_name == current_group_name

        group_current_state = get_group_state(current_state.env_state, str(group_name))
        group_next_state_value = next_state_values[group_name]

        if is_current_group:
            group_action = action
        else:
            group_action = 0

        group_next_state, is_group_terminal, is_group_restart = build_group_next_state(config,
                                                                                       group_next_state_value,
                                                                                       group_current_state,
                                                                                       env_restart_steps,
                                                                                       env_steps_from_alerts,
                                                                                       group_action,
                                                                                       group_name,
                                                                                       is_current_group)

        if is_current_group:
            is_group_next_terminal = is_group_terminal
            is_group_next_restart = is_group_restart

        next_state.env_state[group_name] = group_next_state

    return next_state, is_group_next_terminal, is_group_next_restart


def get_reward_from_env(env_name,
                        config,
                        group_name,
                        next_state_terminal,
                        current_state,
                        env_steps_from_alert,
                        env_restart_steps,
                        action: int,
                        ) \
        -> Union[Dict, float]:
    assert env_name in ["simulation", "real"]

    group_current_state = get_group_state(current_state.env_state, group_name)

    is_group_current_state_restart = is_state_restart(group_current_state.restart_steps,
                                                      env_restart_steps)
    is_group_current_state_terminal = is_state_terminal(config,
                                                        group_name,
                                                        group_current_state.value,
                                                        is_group_current_state_restart)

    if is_group_current_state_restart or is_group_current_state_terminal:
        return 0

    good_alert, false_alert, missed_alert = get_reward_type_for_group(group_current_state.steps_from_alert,
                                                                      env_steps_from_alert,
                                                                      next_state_terminal,
                                                                      action)

    reward = calc_reward(config,
                         good_alert,
                         false_alert,
                         missed_alert,
                         group_current_state.steps_from_alert,
                         env_steps_from_alert)

    return reward


def get_reward_for_alert_from_prediction(config,
                                         group_name,
                                         prediction,
                                         env_steps_from_alert):
    group_prediction = prediction[group_name]
    prediction_quantile = get_num_quantiles() // 2
    lb, ub = get_group_lower_and_upper_bounds(config, group_name)
    for step_from_alert in range(env_steps_from_alert):
        step_group_prediction = group_prediction[step_from_alert][prediction_quantile]
        if step_group_prediction < lb or step_group_prediction > ub:
            reward_good_alert = get_good_alert_reward(config)
            return calc_good_alert_reward(step_from_alert, env_steps_from_alert, reward_good_alert)

    return get_false_alert_reward(config)


def get_reward_type_for_group(group_current_steps_from_alert: int,
                              env_steps_from_alert: int,
                              is_group_next_state_terminal: bool,
                              action: int):
    good_alert = False
    false_alert = False
    missed_alert = False

    if is_missed_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                       action):
        missed_alert = True
    elif is_false_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                        action):
        false_alert = True
    elif is_good_alert(is_group_next_state_terminal, group_current_steps_from_alert, env_steps_from_alert,
                       action):
        good_alert = True
    else:
        pass
    return good_alert, false_alert, missed_alert


def calc_reward(config: dict,
                good_alert: bool,
                false_alert: bool,
                missed_alert: bool,
                current_steps_from_alert: int,
                env_steps_from_alert: int):
    assert os.getenv("REWARD_TYPE") in ["CheapFP", "ExpensiveFP"]

    reward_false_alert = get_false_alert_reward(config)
    reward_missed_alert = get_missed_alert_reward(config)
    reward_good_alert = get_good_alert_reward(config)

    if good_alert:
        return calc_good_alert_reward(current_steps_from_alert, env_steps_from_alert, reward_good_alert)
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


def is_good_alert(next_state_terminal, current_steps_from_alert, env_steps_from_alert, action):
    if next_state_terminal and ((current_steps_from_alert < env_steps_from_alert and action == 0) or
                                (current_steps_from_alert == env_steps_from_alert and action == 1)):
        return True
    return False


def calc_good_alert_reward(current_steps_from_alert, env_steps_from_alert, reward_good_alert):
    return reward_good_alert * (env_steps_from_alert - (current_steps_from_alert - 1))


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

                if next_steps_from_alert == 0:
                    next_steps_from_alert = env_steps_from_alert

    return next_steps_from_alert


def is_state_terminal(config, group_name, value, current_state_restart=None):
    is_terminal = False
    if not current_state_restart:
        is_terminal = is_out_of_bounds(config, group_name, value)
    return is_terminal


def is_out_of_bounds(config, group_name, value):
    lb, ub = get_group_lower_and_upper_bounds(config, group_name)
    if value < lb or value > ub:
        return True
    return False


def update_restart_steps(is_group_next_state_terminal: bool, current_restart_steps: int, env_restart_steps: int):
    next_restart_steps = current_restart_steps
    if is_group_next_state_terminal:
        next_restart_steps = env_restart_steps - 1 if env_restart_steps > 1 else env_restart_steps

    else:
        if is_state_restart(current_restart_steps, env_restart_steps):
            if current_restart_steps == 1:
                next_restart_steps = env_restart_steps
            else:
                next_restart_steps = current_restart_steps - 1

    return next_restart_steps


def get_group_names_from_df(config, df):
    return pd.unique(df[config.get("GroupKeyword")])


def is_state_restart(current_restart_steps, env_restart_steps):
    return current_restart_steps < env_restart_steps


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


def get_num_series(config, df):
    return len(list(df[config.get("GroupKeyword")].unique()))


def build_state_from_df_time_idx(config, df, time_idx):
    current_env_state = {}
    last_sample_df = df[df['time_idx'] == time_idx]
    for idx, sample in last_sample_df.iterrows():
        group_state = State(sample[config.get("GroupKeyword")],
                            get_env_steps_from_alert(config),
                            get_env_restart_steps(config),
                            sample[config.get("ValueKeyword")],
                            [])
        current_env_state[sample[config.get("GroupKeyword")]] = group_state
    return current_env_state


def set_env_state(env, state):
    env.current_state = state


def set_env_chosen_quantile(env, chosen_quantile):
    env.env_group_quantile_idx = chosen_quantile


def set_env_group(env, group_name):
    env.env_group_name = group_name
