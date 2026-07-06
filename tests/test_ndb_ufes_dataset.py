import numpy as np
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms

from src.dataset.ndb_ufes_dataset import NDBUfesDataset


def test_dataset_converts_augmented_numpy_image_before_transform(tmp_path):
    image_path = tmp_path / "patch.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)
    dataset = NDBUfesDataset(
        images=np.array([image_path]),
        labels=np.array([1]),
        classes_dict={"x": 1},
        aug=iaa.Resize({"height": 8, "width": 8}),
        transform=transforms.Compose([
            transforms.Resize((4, 4)),
            transforms.ToTensor(),
        ]),
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 4, 4)
    assert label == 1
