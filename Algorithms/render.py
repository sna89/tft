import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
import os
from config import get_task_folder_name, REGRESSION_TASK_TYPE, PLOT
from EnvCommon.env_thts_common import get_last_val_time_idx


def add_group_y_value_plot(config, fig, test_df, group_name, time_idx_list):
    group_y = list(test_df[(test_df[config.get("GroupKeyword")] == group_name)
                           & (test_df.time_idx.isin(time_idx_list))]
                   [config.get("ValueKeyword")].values)
    fig.add_trace(
        go.Scatter(x=time_idx_list, y=group_y, name="Group: {}".format(group_name),
                   line=dict(color='royalblue', width=1))
    )
    return fig


def add_group_step_decision_to_plot(config,
                                    fig,
                                    test_df,
                                    group_name,
                                    time_idx,
                                    reward,
                                    action,
                                    terminal,
                                    restart,
                                    steps_from_alert,
                                    restart_steps):
    y_value = list(
        test_df[(test_df[config.get("GroupKeyword")] == group_name) & (test_df.time_idx == time_idx)]
        [config.get("ValueKeyword")].values)

    fig.add_trace(
        go.Scatter(x=[time_idx],
                   y=y_value,
                   hovertext="StepsFromAlert: {}, \n"
                             "RestartSteps: {}, \n"
                             "Action: {}, \n"
                             "Reward: {}, \n"
                             "Terminal: {}, \n"
                             "Restart: {}"
                             "".format(steps_from_alert,
                                       restart_steps,
                                       action,
                                       reward,
                                       terminal,
                                       restart),
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                       color="orange" if restart else
                       "purple" if terminal else
                       'green' if not action
                       else "red"
                   )
                   )
    )
    return fig


def render(config,
           env_group_name,
           test_df: pd.DataFrame(),
           action_history: List[int],
           reward_history: List[float],
           terminal_history: List[bool],
           restart_history: List[bool],
           alert_prediction_steps_history: List[int],
           restart_steps_history: List[int],
           task_type=REGRESSION_TASK_TYPE):

    min_test_time_idx = get_last_val_time_idx(config, test_df)
    time_idx_list = list(test_df[test_df.time_idx >= min_test_time_idx]['time_idx'].unique())

    fig = go.Figure()
    fig = add_group_y_value_plot(config, fig, test_df, env_group_name, time_idx_list)

    current_time_idx_list = list(time_idx_list[:len(action_history)])
    for idx, time_idx in enumerate(current_time_idx_list):
        reward = reward_history[idx]
        action = action_history[idx]
        terminal = terminal_history[idx]
        restart = restart_history[idx]
        steps_from_alert = alert_prediction_steps_history[idx]
        restart_steps = restart_steps_history[idx]

        fig = add_group_step_decision_to_plot(config,
                                              fig,
                                              test_df,
                                              env_group_name,
                                              time_idx,
                                              reward,
                                              action,
                                              terminal,
                                              restart,
                                              steps_from_alert,
                                              restart_steps)

    fig.update_xaxes(title_text="<b>time_idx</b>")
    fig.update_yaxes(title_text="<b>Actual</b>")

    folder_name = get_task_folder_name(task_type)
    fig.write_html(os.path.join(PLOT,
                                os.getenv("DATASET"),
                                folder_name,
                                'render_synthetic_{}.html'.format(env_group_name)))