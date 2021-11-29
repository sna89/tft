import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
import os
from config import PLOT
from env_thts_common import get_last_val_time_idx


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
           group_names,
           test_df: pd.DataFrame(),
           run_time: float,
           action_history: List[Dict],
           reward_history: List[Dict],
           terminal_history: List[Dict],
           restart_history: List[Dict],
           alert_prediction_steps_history: List[Dict],
           restart_steps_history: List[Dict]):

    min_test_time_idx = get_last_val_time_idx(config, test_df)
    time_idx_list = list(test_df[test_df.time_idx >= min_test_time_idx]['time_idx'].unique())

    for group_name in group_names:
        fig = go.Figure()
        fig = add_group_y_value_plot(config, fig, test_df, group_name, time_idx_list)

        current_time_idx_list = list(time_idx_list[:len(action_history)])
        for idx, time_idx in enumerate(current_time_idx_list):
            reward = reward_history[idx][group_name]
            action = action_history[idx][group_name]
            terminal = terminal_history[idx][group_name]
            restart = restart_history[idx][group_name]
            steps_from_alert = alert_prediction_steps_history[idx][group_name]
            restart_steps = restart_steps_history[idx][group_name]

            fig = add_group_step_decision_to_plot(config,
                                                  fig,
                                                  test_df,
                                                  group_name,
                                                  time_idx,
                                                  reward,
                                                  action,
                                                  terminal,
                                                  restart,
                                                  steps_from_alert,
                                                  restart_steps)

        fig.update_xaxes(title_text="<b>time_idx</b>")
        fig.update_yaxes(title_text="<b>Actual</b>")
        fig.write_html(os.path.join(PLOT, 'render_synthetic_{}.html'.format(group_name)))