import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
import os
from utils import PLOT
from config import DATETIME_COLUMN


def get_min_test_time_idx(config, test_df):
    return test_df.time_idx.min() + config.get("Data").get("EncoderLength") + config.get("Data").get("PredictionLength")


def add_group_y_value_plot(config, fig, test_df, group_name, dt_list):
    group_y = list(test_df[(test_df[config.get("Data").get("GroupKeyword")] == group_name)
                           & (test_df[DATETIME_COLUMN].isin(dt_list))]
                   [config.get("Data").get("ValueKeyword")].values)
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(dt_list), y=group_y, name="Group: {}".format(group_name),
                   line=dict(color='royalblue', width=1))
    )
    return fig


def add_group_step_decision_to_plot(config,
                                    fig,
                                    test_df,
                                    group_name,
                                    dt,
                                    reward,
                                    anomaly,
                                    action,
                                    terminal,
                                    restart,
                                    steps_from_alert,
                                    restart_steps):
    y_value = list(
        test_df[(test_df[config.get("Data").get("GroupKeyword")] == group_name) & (test_df[DATETIME_COLUMN] == dt)]
        [config.get("Data").get("ValueKeyword")].values)

    fig.add_trace(
        go.Scatter(x=[pd.to_datetime(dt)],
                   y=y_value,
                   hovertext="StepsFromAlert: {}, \n"
                             "RestartSteps: {}, \n"
                             "Anomaly: {}, \n"
                             "Action: {}, \n"
                             "Reward: {}, \n"
                             "Terminal: {}, \n"
                             "Restart: {}"
                             "".format(steps_from_alert,
                                       restart_steps,
                                       anomaly,
                                       action,
                                       reward,
                                       terminal,
                                       restart),
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                       color="yellow" if anomaly else
                       "orange" if restart else
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
           action_history: List[Dict],
           anomaly_history: List[Dict],
           reward_history: List[Dict],
           terminal_history: List[Dict],
           restart_history: List[Dict],
           alert_prediction_steps_history: List[Dict],
           restart_steps_history: List[Dict]):

    min_test_time_idx = get_min_test_time_idx(config, test_df)
    dt_list = list(test_df[test_df.time_idx >= min_test_time_idx][DATETIME_COLUMN].unique())

    for group_name in group_names:
        fig = go.Figure()
        fig = add_group_y_value_plot(config, fig, test_df, group_name, dt_list)

        current_dt_list = list(dt_list[:len(action_history)])
        for idx, dt in enumerate(current_dt_list):
            reward = reward_history[idx][group_name]
            anomaly = anomaly_history[idx][group_name]
            action = action_history[idx][group_name]
            terminal = terminal_history[idx][group_name]
            restart = restart_history[idx][group_name]
            steps_from_alert = alert_prediction_steps_history[idx][group_name]
            restart_steps = restart_steps_history[idx][group_name]

            fig = add_group_step_decision_to_plot(config,
                                                  fig,
                                                  test_df,
                                                  group_name,
                                                  dt,
                                                  reward,
                                                  anomaly,
                                                  action,
                                                  terminal,
                                                  restart,
                                                  steps_from_alert,
                                                  restart_steps)

        fig.update_xaxes(title_text="<b>Time</b>")
        fig.update_yaxes(title_text="<b>Actual</b>")
        fig.write_html(os.path.join(PLOT, 'render_synthetic_{}.html'.format(group_name)))
