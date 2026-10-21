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
    import imgaug

    imgaug.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """For the processes of a DataLoader. Each worker has its own copy of the augmenter and of its random generator, so without this every worker would draw the
    same augmentations: the augmenter of the dataset is seeded from the seed PyTorch gives the worker, which comes from the seeded generator of the loader, so the
    same seed gives the same augmentations. The libraries are also kept to one thread in each worker: many workers with many threads each slow down the machine."""
    import cv2
    from imgaug import augmenters as iaa

    iaa.AddToHueAndSaturation()  # a worker that is started afresh (the default on a Mac) has not run the setup that fills the colour table of this augmenter
    augmenter = getattr(torch.utils.data.get_worker_info().dataset, "aug", None)
    if augmenter is not None:
        augmenter.seed_(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
