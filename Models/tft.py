from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import TimeSeriesDataSet
from Utils.utils import save_to_pickle


def optimize_tft_hp(train_dl, val_dl, study_pkl_path, study_path, loss):
    study = optimize_hyperparameters(
        train_dl,
        val_dl,
        model_path=study_path,
        n_trials=10,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
        loss=loss
    )
    save_to_pickle(study, study_pkl_path)
    return study


def create_tft_model(training_data: TimeSeriesDataSet, loss, output_size, study=None):
    if study:
        params = study.best_params
        del params['gradient_clip_val']

        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            params,
            hidden_size=params['hidden_size'],
            output_size=output_size,
            loss=loss,
        )

    else:
        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.005,
            hidden_size=128,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=4,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=32,  # set to <= hidden_size
            output_size=output_size,
            loss=loss,
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=2,
            log_interval=1
        )
    return tft
