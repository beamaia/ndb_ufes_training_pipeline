from abc import ABC

from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

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
            return SGD(model_params, self.learning_rate, self. momentum)
        else:
            raise ValueError(f"String value '{self.optimizer_name}' invalid for chosing optimizer.")
        
    def _select_scheduler(self):
        if self.scheduler_name == "step_lr":
            step_size = self.scheduler_params.get('step_size', 30)
            gamma = self.scheduler_params.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"String value '{self.scheduler_name}' invalid for chosing scheduler.")