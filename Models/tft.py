import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import TimeSeriesDataSet
import pickle
import os


def create_trainer():
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
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


def optimize_tft_hp(config, train_dl, val_dl):
    study_full_path = os.path.join(config.get("StudyPath"), "study.pkl")

    if not os.path.isfile(study_full_path):
        study = optimize_hyperparameters(
            train_dl,
            val_dl,
            model_path=config.get("StudyPath"),
            n_trials=25,
            max_epochs=10,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )

        with open(study_full_path, "wb") as f:
            pickle.dump(study, f)

    else:
        with open(study_full_path, "rb") as f:
            study = pickle.load(f)

    return study


def create_tft_model(training_data: TimeSeriesDataSet, study=None):
    if study:
        params = study.best_params
        del params['gradient_clip_val']

        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            params,
            output_size=5,
            loss=QuantileLoss([0.1, 0.3, 0.5, 0.7, 0.9]),
        )
    else:
        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.0001,
            hidden_size=128,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=2,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=16,  # set to <= hidden_size
            output_size=5,
            loss=QuantileLoss([0.1, 0.3, 0.5, 0.7, 0.9]),
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


def get_model_from_trainer(trainer):
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = get_model_from_checkpoint(best_model_path)
    return best_tft


def get_model_from_checkpoint(checkpoint):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    return best_tft

