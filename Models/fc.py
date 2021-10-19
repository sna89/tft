import torch
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from torch import nn
from typing import Dict
from pytorch_forecasting.models import BaseModel
from Loss.weighted_cross_entropy import WeightedCrossEntropy
import os
from pytorch_forecasting import TemporalFusionTransformer


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, dropout: float = 0.1):
        super().__init__()

        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)])
        module_list.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class FullyConnectedModel(BaseModel):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 n_classes: int,
                 dropout: float,
                 loss=WeightedCrossEntropy(),
                 learning_rate=1e-3,
                 **kwargs,
                 ):
        self.save_hyperparameters()
        super().__init__(**kwargs)
        self.loss = loss

        checkpoint = os.getenv("CHECKPOINT_REG")
        self.fitted_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)

        layers = list(self.fitted_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.mlp = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size * self.hparams.n_classes,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
            dropout=dropout
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = x["encoder_cont"].size(0)

        # self.feature_extractor.eval()
        # with torch.no_grad():
        representation = self.fitted_model(x, True)

        prediction = representation['prediction']
        prediction = self.mlp(prediction)
        prediction = prediction.unsqueeze(-1).view(batch_size, -1, self.hparams.n_classes)
        prediction = self.softmax(prediction)
        return self.to_network_output(prediction=prediction)


