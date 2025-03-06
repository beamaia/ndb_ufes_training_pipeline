from abc import ABC

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import SGD, Adam

class OptimizerSchedulerSelector(ABC):
    def __init__(self, optimizer, 
                 learning_rate,
                 momentum, 
                 model_params,
                 scheduler, 
                 optimizer_params=None,
                 scheduler_params=None):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_params = optimizer_params

        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params

        self.optimizer = self._select_optimizer(model_params)
        self.scheduler = self._select_scheduler()

    def _select_optimizer(self, model_params):
        if self.optimizer_name == "sgd":
            return SGD(model_params, self.learning_rate, self.momentum)
        elif self.optimizer_name=="adam":
            return Adam(model_params, self.learning_rate)
        else:
            raise ValueError(f"String value '{self.optimizer_name}' invalid for chosing optimizer.")
        
    def _select_scheduler(self):
        if self.scheduler_name == "step_lr":
            step_size = self.scheduler_params.get('step_size', 30)
            gamma = self.scheduler_params.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_name == "reduce_lr_on_plateau":
            mode = self.scheduler_params.get('mode', "min")
            factor = self.scheduler_params.get('factor', 0.1)
            patience = self.scheduler_params.get("patience", 10)
            min_lr = self.scheduler.params.get("min_lr", 0)
            return ReduceLROnPlateau(self.optimizer)
        else:
            raise ValueError(f"String value '{self.scheduler_name}' invalid for chosing scheduler.")