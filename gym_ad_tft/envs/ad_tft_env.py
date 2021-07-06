import gym
from gym import spaces
import numpy as np


class AdTftEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, tft_model):
        self.config = config
        self.tft_model = tft_model

        self.reward_false_alert = config["Rewards"]["FalseAlert"]
        self.reward_missed_alert = config["Rewards"]["MissedAlert"]
        self.reward_good_alert = config["Rewards"]["GoodAlert"]

        self.alert_prediction_steps = config["Env"]["AlertPredictionSteps"]
        self.anomaly_bounds = config["AnomalyConfig"]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=1)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass