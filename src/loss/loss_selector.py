from abc import ABC

from torch.nn import CrossEntropyLoss

class LossSelector(ABC):
    def __init__(self, loss, use_weight, weights=None):
        self.loss_name = loss
        self.use_weight = use_weight
        self.weights = weights

        self.loss = self._select_loss()

    def _select_loss(self):

        if self.loss_name == "cross_entropy":
            return CrossEntropyLoss(weight=self.weights) if self.use_weight and self.weights is not None else CrossEntropyLoss()
        else:
            raise ValueError(f"String value {self.loss_name} invalid for choosing loss function.")
