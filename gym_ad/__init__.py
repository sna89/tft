from gym.envs.registration import register
import os
from config import get_config
from Models.trainer import get_model_from_checkpoint
from utils import load_pickle

config = get_config(os.getenv("DATASET"))
forecasting_model = get_model_from_checkpoint(os.getenv("CHECKPOINT_REG"), os.getenv("MODEL_NAME"))
test_df = load_pickle(config.get("TestDataFramePicklePath"))

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdEnv',
    kwargs={"config": config,
            "forecasting_model": forecasting_model,
            "test_df": test_df}
)
