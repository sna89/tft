from EnvCommon.env_thts_common import get_reward, build_next_state, EnvState, get_last_val_time_idx, \
    get_env_steps_from_alert, get_env_restart_steps, \
    get_group_names_from_df, get_num_series, build_state_from_df_time_idx
from data_utils import get_group_id_group_name_mapping

NUM_ACTIONS = 2


class AdEnv:
    def __init__(self, config, forecasting_model, test_df, test_ts_ds):
        self.env_name = "simulation"
        self.config = config
        self.forecasting_model = forecasting_model
        self.test_df = test_df
        self.test_ts_ds = test_ts_ds

        self.group_id_group_name_mapping = get_group_id_group_name_mapping(self.config, self.test_ts_ds)
        self.group_names = get_group_names_from_df(config, test_df)

        self.env_steps_from_alert = get_env_steps_from_alert(self.config)
        self.env_restart_steps = get_env_restart_steps(self.config)
        self._env_group_name = None

        self.num_series = get_num_series(self.config, self.test_df)
        self.action_space = NUM_ACTIONS

        self._current_state = EnvState()
        self.reset()

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, current_state):
        self._current_state = current_state

    @property
    def env_group_name(self):
        return self._env_group_name

    @env_group_name.setter
    def env_group_name(self, env_group_name):
        self._env_group_name = env_group_name

    def step(self, action: int, group_name_prediction_mapping):
        assert action in [0, 1], "Action must be part of action space"
        assert (all(series_state.steps_from_alert < self.env_steps_from_alert
                    for _, series_state in self.current_state.env_state.items())
                and action == 0) or any((series_state.steps_from_alert == self.env_steps_from_alert
                                         for _, series_state in self.current_state.env_state.items()))

        next_state, is_next_state_terminal, _ = build_next_state(self.env_name,
                                                                 self.config,
                                                                 self.current_state,
                                                                 self.env_group_name,
                                                                 self.group_names,
                                                                 group_name_prediction_mapping,
                                                                 self.env_steps_from_alert,
                                                                 self.env_restart_steps,
                                                                 action)

        reward = get_reward(self.env_name,
                            self.config,
                            self.env_group_name,
                            is_next_state_terminal,
                            self.current_state,
                            self.env_steps_from_alert,
                            self.env_restart_steps,
                            action)

        return next_state, reward

    def reset(self):
        self.current_state.env_state.clear()
        last_val_time_idx = get_last_val_time_idx(self.config, self.test_df)
        self.current_state.env_state = build_state_from_df_time_idx(self.config, self.test_df, last_val_time_idx)
        return self.current_state
