import pandas as pd
from env_thts_common import get_num_iterations, get_group_idx_mapping, get_group_names, get_group_lower_and_upper_bounds, \
    is_group_prediction_out_of_bound
import os


class TrajectorySample:
    def __init__(self, config, deepar_model, val_df, test_df, num_trajectories=1000):
        assert os.getenv("MODEL_NAME") == "DeepAR", "Must use deep ar model to run TrajectorySample algorithm"

        self.config = config
        self.enc_len = self.config.get("EncoderLength")
        self.pred_len = self.config.get("PredictionLength")
        self.actions = [0, 1]
        self.deepar_model = deepar_model
        self.val_df = val_df
        self.test_df = test_df
        self.num_trajectories = num_trajectories
        self.group_idx_mapping = get_group_idx_mapping(self.config, self.deepar_model, test_df)
        self.rollout_policy = self._define_rollout_policy()

    def run(self):
        num_iterations = get_num_iterations(self.test_df, self.enc_len)
        for iteration in range(1, num_iterations):
            prediction_df = self._build_prediction_df(self.val_df, self.test_df, iteration)
            prediction, x = self.deepar_model.predict(prediction_df, mode="prediction", return_x=True, n_samples=self.num_trajectories)
            action = self._choose_action(prediction, x)

    def _choose_action(self, prediction, x):
        scores = []
        group_names = get_group_names(self.group_idx_mapping)
        for action in self.actions:
            score = 0
            for group_name in group_names:
                group_prediction = prediction[self.group_idx_mapping[group_name]]
                lb = self.rollout_policy[group_name]['lb']
                ub = self.rollout_policy[group_name]['ub']
                out_of_bound = is_group_prediction_out_of_bound(group_prediction, lb, ub)

    def _define_rollout_policy(self):
        policy = {}
        group_names = get_group_names(self.group_idx_mapping)
        for group_name in group_names:
            policy[group_name] = {}
            lb, ub = get_group_lower_and_upper_bounds(self.config, group_name)
            policy[group_name]['lb'] = lb + 0.3
            policy[group_name]['ub'] = ub - 0.3
        return policy

    def _build_prediction_df(self, val_df, test_df, iteration):
        start_time_idx_test_df = val_df.time_idx.max() + 1
        end_time_idx_test_df = start_time_idx_test_df + iteration
        time_idx_range = list(range(start_time_idx_test_df, end_time_idx_test_df))
        prediction_df = pd.concat([val_df, test_df[test_df.time_idx.isin(time_idx_range)]], axis=0)
        prediction_df.reset_index(drop=True, inplace=True)
        return prediction_df
