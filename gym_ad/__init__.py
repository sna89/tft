from gym.envs.registration import register
from config import get_config
from Models.trainer import get_model_from_checkpoint
from utils import load_pickle
import os

config = get_config()

model_path = os.getenv("CHECKPOINT") if os.getenv("CHECKPOINT") else config.get("load_model_path")
model = get_model_from_checkpoint(model_path, config.get("model"))

val_df = load_pickle(config.get("val_df_pkl_path"))
test_df = load_pickle(config.get("test_df_pkl_path"))

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdEnv',
    kwargs={"config": config,
            "model": model,
            "val_df": val_df,
            "test_df": test_df}
)
