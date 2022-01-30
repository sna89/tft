from Algorithms.thts.node import DecisionNode, ChanceNode
import numpy as np
from EnvCommon.env_thts_common import is_out_of_bounds, calc_good_alert_reward
from config import get_missed_alert_reward, \
    get_good_alert_reward, \
    get_false_alert_reward, \
    get_env_steps_from_alert, \
    QUANTILES, \
    get_num_quantiles
from data_utils import get_group_lower_and_upper_bounds

UCT_BIAS = np.sqrt(2)


def calc_out_of_lb_probability_estimation(lb, group_prediction_step, q_idx):
    return ((lb - group_prediction_step[q_idx]) /
            (group_prediction_step[q_idx + 1] - group_prediction_step[q_idx])) * \
           (QUANTILES[q_idx + 1] - QUANTILES[q_idx]) + \
           (QUANTILES[q_idx])


def calc_out_of_ub_probability_estimation(ub, group_prediction_step, q_idx):
    return ((group_prediction_step[q_idx + 1] - ub) /
            (group_prediction_step[q_idx + 1] - group_prediction_step[q_idx])) * \
           (QUANTILES[q_idx + 1] - QUANTILES[q_idx]) + \
           (1 - QUANTILES[q_idx + 1])


def estimate_out_of_bound_step_probability(lb, ub, group_prediction_step):
    num_quantiles = get_num_quantiles()
    lq, uq = group_prediction_step[0], group_prediction_step[-1]
    if lb < lq and uq < ub:
        return 0
    elif ub < lq or lb > uq:
        return 1
    else:
        for q_idx in range(num_quantiles - 1):
            if group_prediction_step[q_idx] <= lb <= group_prediction_step[q_idx + 1]:
                return calc_out_of_lb_probability_estimation(lb, group_prediction_step, q_idx)

            elif group_prediction_step[q_idx] <= ub <= group_prediction_step[q_idx + 1]:
                return calc_out_of_ub_probability_estimation(ub, group_prediction_step, q_idx)


def get_prediction_quantile(group_prediction):
    quantiles_sample = group_prediction[0]
    prediction_quantile = len(quantiles_sample) // 2
    return prediction_quantile


def run_heuristic_wait(config, group_name, steps, group_prediction):
    missed_alert_reward = get_missed_alert_reward(config)
    lb, ub = get_group_lower_and_upper_bounds(config, group_name)

    values = []
    out_of_bound_probabilities = []
    for step in range(steps):
        group_prediction_step = group_prediction[step]
        out_of_bound_step_p_estimation = estimate_out_of_bound_step_probability(lb,
                                                                                ub,
                                                                                group_prediction_step)
        out_of_bound_p_estimation = estimate_out_of_bound_probability(out_of_bound_probabilities,
                                                                      out_of_bound_step_p_estimation)
        out_of_bound_probabilities.append(out_of_bound_step_p_estimation)
        step_value = out_of_bound_p_estimation * missed_alert_reward
        values.append(step_value)

        if out_of_bound_p_estimation == 1:
            break

    value = sum(values)
    return value


def run_heuristic_action(config, group_name, steps, group_prediction):
    good_alert_reward = get_good_alert_reward(config)
    false_alert_reward = get_false_alert_reward(config)
    env_steps_from_alert = get_env_steps_from_alert(config)
    lb, ub = get_group_lower_and_upper_bounds(config, group_name)

    values = []
    out_of_bound_probabilities = []
    for step in range(env_steps_from_alert):
        group_prediction_step = group_prediction[step]
        out_of_bound_step_p_estimation = estimate_out_of_bound_step_probability(lb,
                                                                                ub,
                                                                                group_prediction_step)
        out_of_bound_p_estimation = estimate_out_of_bound_probability(out_of_bound_probabilities,
                                                                      out_of_bound_step_p_estimation)
        out_of_bound_probabilities.append(out_of_bound_step_p_estimation)

        step_good_alert_reward = calc_good_alert_reward(step, env_steps_from_alert, good_alert_reward)

        if step < env_steps_from_alert - 1:
            step_value = out_of_bound_p_estimation * step_good_alert_reward

        elif step == env_steps_from_alert - 1:
            step_value = out_of_bound_p_estimation * step_good_alert_reward + \
                         (1 - out_of_bound_p_estimation) * false_alert_reward

        values.append(step_value)

        if out_of_bound_p_estimation == 1:
            break

    if steps > env_steps_from_alert:
        remaining_steps = steps - env_steps_from_alert
        group_prediction = group_prediction[env_steps_from_alert:]
        wait_value = run_heuristic_wait(config, group_name, remaining_steps, group_prediction)
        values.append(wait_value)
        values.extend([0] * (remaining_steps - 1))

    value = sum(values)
    return value


def estimate_out_of_bound_probability(probabilities, current_probability):
    out_of_bound_probability = 1
    for p in probabilities:
        out_of_bound_probability *= (1 - p)
    out_of_bound_probability *= current_probability
    return out_of_bound_probability


def run_heuristic(config, chance_node, steps, prediction, group_name):
    group_prediction = prediction[group_name]
    value = 0

    if chance_node.action == 0:
        value = run_heuristic_wait(config, group_name, steps, group_prediction)

    elif chance_node.action == 1:
        value = run_heuristic_action(config, group_name, steps, group_prediction)

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
    q_value = node.reward + nominator / float(denominator) if denominator > 0 else node.reward
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
