import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every source of randomness the training uses, so the same seed gives the same run on the same hardware.

    Different hardware (for example a Mac and an NVIDIA GPU) can still differ in the last digits.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPUs
    import imgaug  # imported here: the numpy compatibility patch of the organizer must be applied first

    imgaug.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
