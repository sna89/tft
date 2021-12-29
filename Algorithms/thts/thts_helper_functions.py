from Algorithms.thts.node import DecisionNode, ChanceNode
import numpy as np
from EnvCommon.env_thts_common import is_out_of_bounds, calc_good_alert_reward
from config import get_missed_alert_reward, \
    get_good_alert_reward, \
    get_false_alert_reward, \
    get_tree_depth, \
    get_env_steps_from_alert, \
    QUANTILES, \
    get_num_quantiles
from data_utils import get_group_lower_and_upper_bounds

UCT_BIAS = np.sqrt(2)


def calc_out_of_bound_probability_estimation(lb, ub, group_prediction_step):
    num_quantiles = get_num_quantiles()
    lq, uq = group_prediction_step[0], group_prediction_step[-1]
    if lb < lq and uq < ub:
        return 0
    elif ub < lq or lb > uq:
        return 1
    else:
        for q_idx in range(num_quantiles - 1):
            if group_prediction_step[q_idx] <= lb <= group_prediction_step[q_idx + 1]:
                return (QUANTILES[q_idx + 1] + QUANTILES[q_idx]) / 2
            elif group_prediction_step[q_idx] <= ub <= group_prediction_step[q_idx + 1]:
                return 1 - (QUANTILES[q_idx + 1] + QUANTILES[q_idx]) / 2


def get_prediction_quantile(group_prediction):
    quantiles_sample = group_prediction[0]
    prediction_quantile = len(quantiles_sample) // 2
    return prediction_quantile


def run_heuristic_wait(config, group_name, steps, prediction_quantile, group_prediction):
    value = 0
    for step in range(steps):
        depth_prediction = group_prediction[step][prediction_quantile]
        if is_out_of_bounds(config, group_name, depth_prediction):
            value = get_missed_alert_reward(config)

    return value


def run_heuristic(config, chance_node: ChanceNode, prediction, group_name):
    group_prediction = prediction[group_name]
    prediction_quantile = get_prediction_quantile(group_prediction)
    tree_depth = get_tree_depth(config)

    value = 0
    if chance_node.action == 0:
        chance_node.value = run_heuristic_wait(config, group_name, tree_depth, prediction_quantile, group_prediction)

    elif chance_node.action == 1:
        good_alert_reward = get_good_alert_reward(config)
        false_alert_reward = get_false_alert_reward(config)
        env_steps_from_alert = get_env_steps_from_alert(config)
        lb, ub = get_group_lower_and_upper_bounds(config, group_name)

        values = []
        for step in range(env_steps_from_alert):
            group_prediction_step = group_prediction[step]
            out_of_bound_probability_estimation = calc_out_of_bound_probability_estimation(lb, ub, group_prediction_step)

            step_good_alert_reward = calc_good_alert_reward(step, env_steps_from_alert, good_alert_reward)
            step_value = out_of_bound_probability_estimation * step_good_alert_reward + \
                         (1 - out_of_bound_probability_estimation) * false_alert_reward

            values.append(step_value)

        if tree_depth > env_steps_from_alert:
            steps = tree_depth - env_steps_from_alert
            group_prediction = group_prediction[env_steps_from_alert:]
            wait_value = run_heuristic_wait(config, group_name, steps, prediction_quantile, group_prediction)
            values.append(wait_value)
            values.extend([0] * (steps - 1))

        value = sum(values) / len(values)

    chance_node.value = value


def uct(node: ChanceNode):
    if node.visits == 0:
        return np.inf

    uct_value = UCT_BIAS * np.sqrt(np.log(node.parent.visits) / node.visits) + node.value
    return uct_value


def backup_decision_full_bellman(node: DecisionNode):
    value = 0
    if not node.terminal:
        value = max([node.value for node in node.successors])
    node.value = value


def backup_chance_monte_carlo(node: ChanceNode):
    nominator = 0
    for decision_node in node.successors:
        nominator += decision_node.visits * decision_node.value

    denominator = node.visits
    q_value = node.reward + nominator / float(denominator)
    node.value = q_value


def backup_chance_partial_bellman(node: ChanceNode):
    nominator = 0
    denominator = 0
    for decision_node in node.successors:
        decision_node_prob = decision_node.prob
        nominator += decision_node_prob * decision_node.value
        denominator += decision_node_prob

    q_value = node.reward + nominator / float(denominator)
    node.value = q_value
