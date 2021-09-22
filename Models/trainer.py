import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.models.deepar import DeepAR
from Models.tft import create_tft_model, optimize_tft_hp
from Models.deep_ar import create_deepar_model, optimize_deepar_hp
from Models.fc import FullyConnectedModel
from data_utils import get_dataloader
from utils import load_pickle


def fit_classification_model(config, fitted_model, train_dl, val_dl):
    hidden_size = config.get("HiddenSize")
    fc_model = FullyConnectedModel(fitted_model=fitted_model,
                                   input_size=hidden_size,
                                   output_size=2,
                                   n_hidden_layers=2,
                                   hidden_size=hidden_size//4)
    trainer = create_trainer(gradient_clip_val=0.01)
    trainer = fit(trainer, fc_model, train_dl, val_dl)
    classification_model = get_model_from_trainer(trainer, "Classification")
    return classification_model


def optimize_hp(config, train_ts_ds, val_ts_ds, model_name):
    study = None

    train_dl = get_dataloader(train_ts_ds, is_train=True, config=config)
    val_dl = get_dataloader(val_ts_ds, is_train=False, config=config)

    study_path = os.path.join(config.get("StudyPath"), "study.pkl")
    if os.path.isfile(study_path) and os.getenv("STUDY") == "False":
        study = load_pickle(study_path)

    else:
        if os.getenv("STUDY") == "True":
            if model_name == "TFT":
                study = optimize_tft_hp(config, train_dl, val_dl, study_path)
            elif model_name == "DeepAR":
                study = optimize_deepar_hp(config, train_dl, val_dl, study_path)

    return study


def fit_regression(config, train_ts_ds, val_ts_ds, model_name, study=None):
    model = None
    if model_name == "TFT":
        model = create_tft_model(train_ts_ds, study)
    elif model_name == "DeepAR":
        model = create_deepar_model(train_ts_ds, None)

    trainer = create_trainer(study)

    if os.getenv("FIT") != "False":
        train_dl = get_dataloader(train_ts_ds, is_train=True, config=config)
        val_dl = get_dataloader(val_ts_ds, is_train=False, config=config)
        trainer = fit(trainer, model, train_dl, val_dl)
        model = get_model_from_trainer(trainer, model_name)
    else:
        checkpoint = os.getenv("CHECKPOINT")
        model = get_model_from_checkpoint(checkpoint, model_name)

    return model


def create_trainer(study=None, gradient_clip_val=0.1):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
    logger = TensorBoardLogger("tb_logs", name="my_model")
    lr_logger = LearningRateMonitor(logging_interval='step')  # log the learning rate

    if study:
        gradient_clip_val = study.best_params['gradient_clip_val']
    else:
        gradient_clip_val = gradient_clip_val

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=200,
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
    elif model_name == "Classification":
        best_model = FullyConnectedModel.load_from_checkpoint(checkpoint)
    return best_model


def get_prediction_mode():
    model_name = os.getenv("MODEL_NAME")
    if model_name == "TFT" :
        mode = "raw"
    elif model_name == "DeepAR":
        mode = "quantiles"
    else:
        raise ValueError
    return mode
