from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR


def create_deepar_model(train_ts_ds):
    deepar = DeepAR.from_dataset(
        train_ts_ds,
        learning_rate=0.0005,
        hidden_size=256,
        dropout=0.1,
        log_interval=10,
        log_val_interval=3,
        loss=NormalDistributionLoss(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    )
    return deepar