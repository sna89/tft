import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import Baseline
import torch
from sklearn.metrics import mean_squared_error
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


def create_trainer():
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min")
    logger = TensorBoardLogger("tb_logs", name="my_model")
    lr_logger = LearningRateMonitor()  # log the learning rate

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=20,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger
    )
    return trainer


def create_tft_model(training_data, train_dl, val_dl):
    study = optimize_hyperparameters(
        train_dl,
        val_dl,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )
    print(study)

    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.005,
        hidden_size=128,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=8,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=32,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
        log_interval=10
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
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return best_tft


def evaluate(trainer, val_dl):
    model = get_fitted_model(trainer)
    actuals = torch.cat([y[0] for x, y in iter(val_dl)])
    predictions = model.predict(val_dl)
    mse = mean_squared_error(actuals, predictions)
    mae = calc_mae(actuals, predictions)
    return mse, mae


def evaluate_base_model(val_dl):
    actuals = torch.cat([y for x, (y, weight) in iter(val_dl)])
    baseline_predictions = Baseline().predict(val_dl)
    mse = mean_squared_error(actuals, baseline_predictions)
    mae = calc_mae(actuals, baseline_predictions)
    return mse, mae


def calc_mae(actuals, predictions):
    mae = (actuals - predictions).abs().mean()
    return mae
