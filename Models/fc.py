import torch
from torch import nn
from typing import Dict
from pytorch_forecasting.models import BaseModel
from Loss.weighted_cross_entropy import WeightedCrossEntropy
import os
from pytorch_forecasting import TemporalFusionTransformer, DeepAR
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule


class FullyConnectedModel(BaseModel):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 n_classes: int,
                 dropout: float,
                 activation_class: str = "ReLU",
                 loss=WeightedCrossEntropy(),
                 learning_rate=1e-3,
                 **kwargs,
                 ):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.loss = loss
        if os.getenv("MODEL_NAME_REG") == "TFT":
            self.fitted_model = TemporalFusionTransformer.load_from_checkpoint(os.getenv("CHECKPOINT_REG"))
        elif os.getenv("MODEL_NAME_REG") == "DeepAR":
            self.fitted_model = DeepAR.load_from_checkpoint(os.getenv("CHECKPOINT_REG"))

        # layers = list(self.fitted_model.children())[:-1]
        # self.feature_extractor = nn.Sequential(*layers)

        self.mlp = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size * self.hparams.n_classes,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
            dropout=dropout,
            activation_class=getattr(nn, self.hparams.activation_class)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = x["encoder_cont"].size(0)

        self.fitted_model.eval()
        with torch.no_grad():
            prediction = self.fitted_model(x, True)
            # prediction = prediction['prediction']

        prediction = self.mlp(prediction)
        prediction = prediction.unsqueeze(-1).view(batch_size, -1, self.hparams.n_classes)
        prediction = self.softmax(prediction)
        return self.to_network_output(prediction=prediction)
