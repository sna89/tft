from gym.envs.registration import register
import os
from config import get_config
from Models.trainer import get_model_from_checkpoint
from utils import load_pickle

config = get_config(os.getenv("DATASET"))
model = get_model_from_checkpoint(os.getenv("CHECKPOINT"), os.getenv("MODEL_NAME"))
val_df = load_pickle(config.get("ValDataFramePicklePath"))
test_df = load_pickle(config.get("TestDataFramePicklePath"))

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdEnv',
    kwargs={"config": config,
            "model": model,
            "val_df": val_df,
            "test_df": test_df}
)
