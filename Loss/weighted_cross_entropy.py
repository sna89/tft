from pytorch_forecasting.metrics import CrossEntropy
import torch.nn.functional as F
import torch


class WeightedCrossEntropy(CrossEntropy):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = None
        if weights:
            self.weights = torch.Tensor(weights)
            if torch.cuda.is_available():
                self.weights = self.weights.cuda(device=torch.device("cuda"))

    def loss(self, y_pred, target):
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), target.view(-1), reduction="none", weight=self.weights).view(
            -1, target.size(-1)
        )
        return loss