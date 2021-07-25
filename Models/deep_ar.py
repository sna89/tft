from pytorch_forecasting.metrics import NormalDistributionLoss
import logging
import os
from typing import Any, Dict, Tuple, Union
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import optuna.logging
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import DeepAR
from pytorch_forecasting.data import TimeSeriesDataSet
import copy
from utils import save_to_pickle, load_pickle


def create_deepar_model(train_ts_ds, study=None):
    if study:
        params = study.best_params
        del params['gradient_clip_val']

        deepar = DeepAR.from_dataset(
            train_ts_ds,
            params,
            log_interval=10,
            log_val_interval=3,
            loss=NormalDistributionLoss(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
        )
    else:
        deepar = DeepAR.from_dataset(
            train_ts_ds,
            learning_rate=0.1071,
            hidden_size=25,
            dropout=0.164,
            log_interval=10,
            log_val_interval=3,
            loss=NormalDistributionLoss(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
        )
    return deepar


def optimize_deepar_hp(config, train_dl, val_dl):
    study_full_path = os.path.join(config.get("StudyPath"), "study.pkl")
    is_study = os.getenv("STUDY") == "True"

    if is_study:
        study = optimize_hyperparameters(
            train_dl,
            val_dl,
            model_path=config.get("StudyPath"),
            n_trials=100,
            max_epochs=20
        )
        save_to_pickle(study, study_full_path)
    else:
        if os.path.isfile(study_full_path):
            study = load_pickle(study_full_path)
        else:
            raise ValueError

    return study


optuna_logger = logging.getLogger("optuna")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def optimize_hyperparameters(
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model_path: str,
        max_epochs: int = 20,
        n_trials: int = 100,
        timeout: float = 3600 * 8.0,  # 8 hours
        gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
        dropout_range: Tuple[float, float] = (0.01, 0.3),
        hidden_size_range: Tuple[int, int] = (16, 2048),
        learning_rate_range: Tuple[float, float] = (1e-5, 1.0),
        trainer_kwargs: Dict[str, Any] = {},
        log_dir: str = "lightning_logs",
        study: optuna.Study = None,
        verbose: Union[int, bool] = None,
        **kwargs,
) -> optuna.Study:
    assert isinstance(train_dataloader.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloader.dataset, TimeSeriesDataSet
    ), "dataloaders must be built from timeseriesdataset"

    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    loss = kwargs.get(
        "loss", NormalDistributionLoss()
    )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next

    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(model_path, "trial_{}".format(trial.number)), filename="{epoch}", monitor="val_loss"
        )

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()
        learning_rate_callback = LearningRateMonitor()
        logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
        gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
        default_trainer_kwargs = dict(
            gpus=[0] if torch.cuda.is_available() else None,
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                metrics_callback,
                learning_rate_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
            logger=logger,
            progress_bar_refresh_rate=[0, 1][optuna_verbose < optuna.logging.INFO],
            weights_summary=[None, "top"][optuna_verbose < optuna.logging.INFO],
        )
        default_trainer_kwargs.update(trainer_kwargs)
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )

        # create model
        hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
        kwargs["loss"] = copy.deepcopy(loss)
        model = DeepAR.from_dataset(
            train_dataloader.dataset,
            dropout=trial.suggest_uniform("dropout", *dropout_range),
            hidden_size=hidden_size,
            log_interval=-1,
            **kwargs,
        )
        model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        return metrics_callback.metrics[-1]["val_loss"].item()

    pruner = optuna.pruners.SuccessiveHalvingPruner()
    if study is None:
        study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
