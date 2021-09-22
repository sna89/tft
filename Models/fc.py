import torch
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from torch import nn
from typing import Dict
from pytorch_forecasting.models import BaseModel
from pytorch_forecasting.metrics import CrossEntropy


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        module_list.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class FullyConnectedModel(BaseModel):
    def __init__(self,
                 fitted_model,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 n_classes: int,
                 loss=CrossEntropy(),
                 **kwargs,
                 ):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.fitted_model = fitted_model

        self.mlp = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size * self.hparams.n_classes,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = x["encoder_cont"].size(0)
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input)
        prediction = prediction.unsqueeze(-1).view(batch_size, -1, self.hparams.n_classes)
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        assert isinstance(dataset.target_normalizer, NaNLabelEncoder), "target normalizer has to encode categories"
        new_kwargs = {
            "n_classes": len(
                dataset.target_normalizer.classes_
            ),
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
                len(dataset.time_varying_known_categoricals) == 0
                and len(dataset.time_varying_known_reals) == 0
                and len(dataset.time_varying_unknown_categoricals) == 0
                and len(dataset.static_categoricals) == 0
                and len(dataset.static_reals) == 0
                and len(dataset.time_varying_unknown_reals) == 1
        ), "Only covariate should be in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)
