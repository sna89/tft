import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from Models.study import get_study_path, get_study_pkl_path
from Models.tft import create_tft_model, optimize_tft_hp
from Models.deep_ar import create_deepar_model, optimize_deepar_hp
from Models.fc_utils import create_mlp_model
from Utils.data_utils import get_dataloader
from Utils.utils import load_pickle, get_model_from_checkpoint, get_model_from_trainer
from config import CLASSIFICATION_TASK_TYPE


def optimize_hp(config, train_ds, val_ds, model_name, task_type, loss=None):
    if task_type == CLASSIFICATION_TASK_TYPE:
        return None

    study_pkl_path = get_study_pkl_path(config, task_type)

    is_study_exists = os.path.isfile(study_pkl_path)
    is_execute_study = os.getenv("STUDY_{}".format(task_type.upper())) == "True"

    if is_study_exists and not is_execute_study:
        try:
            return load_pickle(study_pkl_path)
        except IOError as e:
            raise e

    else:
        study_path = get_study_path(config, task_type)

        train_dl = get_dataloader(train_ds, is_train=True, config=config)
        val_dl = get_dataloader(val_ds, is_train=False, config=config)

        if model_name == "TFT":
            study = optimize_tft_hp(train_dl, val_dl, study_pkl_path, study_path, loss)
        elif model_name == "DeepAR":
            study = optimize_deepar_hp(train_dl, val_dl, study_pkl_path, study_path)
        else:
            raise IOError

    return study


def fit_model(config, task_type, train_ds, val_ds, model_name, loss, output_size, study=None):
    to_fit = os.getenv("FIT_{}".format(task_type.upper())) == "True"

    if to_fit:
        train_dl = get_dataloader(train_ds, is_train=True, config=config)
        val_dl = get_dataloader(val_ds, is_train=False, config=config)

        trainer_model_name = os.getenv("DATASET") + "_" + model_name + "_" + task_type
        trainer = create_trainer(study,
                                 gradient_clip_val=0.1,
                                 model_name=trainer_model_name)
        model = create_model(model_name, train_ds, loss, output_size, study)
        trainer = fit(trainer, model, train_dl, val_dl)

        model = get_model_from_trainer(trainer, model_name, task_type)

    else:
        checkpoint = os.getenv("CHECKPOINT_{}".format(task_type.upper()))
        model = get_model_from_checkpoint(checkpoint, model_name)

    return model


def create_trainer(study=None, gradient_clip_val=0.1, model_name="my_model"):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=3, verbose=True, mode="min")
    logger = TensorBoardLogger("tb_logs", name=model_name)
    lr_logger = LearningRateMonitor(logging_interval='step')  # log the learning rate

    if study:
        gradient_clip_val = study.best_params['gradient_clip_val']
    else:
        gradient_clip_val = gradient_clip_val

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
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


def create_model(model_name, train_ds, loss, output_size, study=None):
    if model_name == "TFT":
        model = create_tft_model(train_ds, loss, output_size, study)
    elif model_name == "DeepAR":
        model = create_deepar_model(train_ds, None)
    elif model_name == "Mlp":
        model = create_mlp_model(loss)
    else:
        raise ValueError
    return model


