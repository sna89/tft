import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.models.deepar import DeepAR


def create_trainer(config, gradient_clip_val=0.1):
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=config.get("Trainer").get("delta"),
                                        patience=config.get("Trainer").get("patience"),
                                        verbose=True,
                                        mode="min")
    logger = TensorBoardLogger("tb_logs", name="my_model")
    lr_logger = LearningRateMonitor(logging_interval='step')  # log the learning rate

    trainer = pl.Trainer(
        gpus=config.get("Trainer").get("gpus"),
        max_epochs=config.get("Trainer").get("epochs"),
        gradient_clip_val=gradient_clip_val,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
        # accelerator="ddp"
    )
    return trainer


def fit(trainer, model, train_dl, val_dl):
    trainer.fit(
        model,
        train_dataloader=train_dl,
        val_dataloaders=val_dl,
    )
    return trainer


def get_model_from_trainer(trainer, model_name):
    best_model_path = trainer.checkpoint_callback.best_model_path
    os.environ["CHECKPOINT"] = best_model_path
    best_tft = get_model_from_checkpoint(best_model_path, model_name)
    return best_tft


def get_model_from_checkpoint(checkpoint, model_name):
    best_model = None
    if model_name == "TFT":
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    elif model_name == "DeepAR":
        best_model = DeepAR.load_from_checkpoint(checkpoint)
    return best_model


def get_prediction_mode(config):
    model_name = config.get("model")
    if model_name == "TFT":
        mode = "raw"
    elif model_name == "DeepAR":
        mode = "quantiles"
    else:
        raise ValueError
    return mode
