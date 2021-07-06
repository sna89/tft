from gym.envs.registration import register
import os
from config import get_config
from Models.tft import get_model_from_checkpoint

config = get_config(os.getenv("DATASET"))
model = get_model_from_checkpoint(os.getenv("CHECKPOINT"))

register(
    id='tft-v0',
    entry_point='gym_ad_tft.envs:AdTftEnv',
    kwargs={"config": config, "tft_model": model}
)
