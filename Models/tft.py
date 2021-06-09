import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


def create_trainer():
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=True, mode="min")
    logger = TensorBoardLogger("tb_logs", name="my_model")
    lr_logger = LearningRateMonitor(logging_interval='step')  # log the learning rate

    trainer = pl.Trainer(
        gpus=2,
        max_epochs=200,
        # gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
        accelerator="ddp"
    )
    return trainer


def create_tft_model(training_data):
    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.0001,
        hidden_size=256,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=3,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=64,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss([0.005, 0.1, 0.25, 0.5, 0.75, 0.9, 0.995]),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=2,
        log_interval=1
    )
    return tft


def fit(trainer, model, train_dl, val_dl):
    trainer.fit(
        model,
        train_dataloader=train_dl,
        val_dataloaders=val_dl,
    )
    return trainer


def get_fitted_model(trainer):
    # best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_path = 'tb_logs/my_model/version_0/checkpoints/epoch=1-step=293.ckpt'
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return best_tft



