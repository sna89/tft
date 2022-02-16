import os
from pytorch_forecasting import TemporalFusionTransformer, DeepAR
from Models.fc import FullyConnectedModel


def create_mlp_model(loss):
    fitted_model = None
    if os.getenv("MODEL_NAME_REG") == "DeepAR":
        fitted_model = DeepAR.load_from_checkpoint(os.getenv("CHECKPOINT_REG"))
    elif os.getenv("MODEL_NAME_REG") == "TFT":
        fitted_model = TemporalFusionTransformer.load_from_checkpoint(os.getenv("CHECKPOINT_REG"))

    input_size = fitted_model.hparams["hidden_size"]
    fc = FullyConnectedModel(input_size=input_size,
                             hidden_size=input_size * 4,
                             output_size=1,
                             n_hidden_layers=1,
                             n_classes=2,
                             dropout=0.5,
                             loss=loss,
                             learning_rate=1e-7)
    return fc
