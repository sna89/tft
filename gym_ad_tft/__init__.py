from gym.envs.registration import register
import os
from config import get_config
from Models.tft import get_model_from_checkpoint
from utils import load_pickle

config = get_config(os.getenv("DATASET"))
model = get_model_from_checkpoint(os.getenv("CHECKPOINT"))
test_ts_ds = load_pickle(config.get("TestTSDatasetPicklePath"))
test_df = load_pickle(config.get("TestDataFramePicklePath"))

register(
    id='tft-v0',
    entry_point='gym_ad_tft.envs:AdTftEnv',
    kwargs={"config": config,
            "tft_model": model,
            "test_ts_ds": test_ts_ds,
            "test_df": test_df}
)
