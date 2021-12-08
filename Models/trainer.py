import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from Models.tft import create_tft_model, optimize_tft_hp
from Models.deep_ar import create_deepar_model, optimize_deepar_hp
from Models.fc import FullyConnectedModel
from data_utils import get_dataloader
from utils import load_pickle
from utils import get_model_from_checkpoint, get_model_from_trainer
from Loss.weighted_cross_entropy import WeightedCrossEntropy


def create_classification_model(reg_study, weights=None):
    input_size = reg_study.best_params['hidden_size'] if reg_study and 'hidden_size' in reg_study.best_params else 64

    fc = FullyConnectedModel(input_size=input_size,
                             hidden_size=input_size // 2,
                             output_size=1,
                             n_hidden_layers=3,
                             n_classes=2,
                             dropout=0.25,
                             loss=WeightedCrossEntropy(weights),
                             learning_rate=0.0001)
    return fc


def fit_classification_model(config, classification_model, train_ts_ds, val_ts_ds):
    classification_checkpoint = os.getenv("CHECKPOINT_CLASS")
    to_fit = os.getenv("FIT_CLASS") == "True"

    if to_fit:
        train_dl = get_dataloader(train_ts_ds, True, config)
        val_dl = get_dataloader(val_ts_ds, False, config)

        trainer = create_trainer(gradient_clip_val=0.01)
        trainer = fit(trainer, classification_model, train_dl, val_dl)
        fitted_classification_model = get_model_from_trainer(trainer, "Classification")

    elif classification_checkpoint and os.path.isfile(classification_checkpoint):
        fitted_classification_model = get_model_from_checkpoint(classification_checkpoint, "Classification")
    else:
        raise ValueError

    return fitted_classification_model


def optimize_hp(config, train_ds, val_ds, model_name, type_='reg'):
    study = None

    train_dl = get_dataloader(train_ds, is_train=True, config=config)
    val_dl = get_dataloader(val_ds, is_train=False, config=config)

    if type_ == "reg":
        study_pkl_path = os.path.join(config.get("StudyRegPath"), "study.pkl")
        if os.path.isfile(study_pkl_path) and os.getenv("STUDY_REG") == "False":
            return load_pickle(study_pkl_path)

    elif type_ == "class":
        study_pkl_path = os.path.join(config.get("StudyClassPath"), "study.pkl")
        if os.path.isfile(study_pkl_path) and os.getenv("STUDY_CLASS") == "False":
            return load_pickle(study_pkl_path)

    else:
        raise ValueError

    if os.getenv("STUDY_REG") == "True" and type_ == "reg":
        study_path = config.get("StudyRegPath")

        if model_name == "TFT":
            study = optimize_tft_hp(train_dl, val_dl, study_pkl_path, study_path)
        elif model_name == "DeepAR":
            study = optimize_deepar_hp(train_dl, val_dl, study_pkl_path, study_path)

    return study


def fit_regression_model(config, train_ds, val_ds, model_name, study=None, type_="reg"):
    model = None
    if model_name == "TFT":
        model = create_tft_model(train_ds, study)
    elif model_name == "DeepAR":
        model = create_deepar_model(train_ds, None)

    trainer = create_trainer(study)

    to_fit = os.getenv("FIT_{}".format(type_.upper()))
    if to_fit == "True":
        train_dl = get_dataloader(train_ds, is_train=True, config=config)
        val_dl = get_dataloader(val_ds, is_train=False, config=config)
        trainer = fit(trainer, model, train_dl, val_dl)
        model = get_model_from_trainer(trainer, model_name)
    elif to_fit == "False":
        checkpoint = os.getenv("CHECKPOINT_{}".format(type_.upper()))
        model = get_model_from_checkpoint(checkpoint, model_name)
    else:
        raise ValueError

    return model


def create_trainer(study=None, gradient_clip_val=0.1):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=3, verbose=True, mode="min")
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


def get_prediction_mode():
    model_name = os.getenv("MODEL_NAME")
    if model_name == "TFT":
        mode = "raw"
    elif model_name == "DeepAR":
        mode = "quantiles"
    else:
        raise ValueError
    return mode
