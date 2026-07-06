import os
from cv2 import COLOR_BGR2RGB, IMREAD_COLOR, cvtColor, imread
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class NDBUfesDataset(Dataset):
    def __init__(self, images, labels, classes_dict, aug = None, transform = None, metadata=None):
        self.images = images
        self.labels = labels
        self.classes = classes_dict
        self.aug = aug
        self.transform = transform
        self.metadata = metadata or []

    def __getitem__(self, index):
        if index >= len(self.images):
            raise IndexError("Index out of bounds in target dataset.")
        
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError(f"File not found in path {self.images[index]}")
        
        image = imread(str(self.images[index]), IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at path {self.images[index]}")
        image = cvtColor(image, COLOR_BGR2RGB)
        label = self.labels[index]

        image = Image.fromarray(image)
        if self.aug:
            image = Image.fromarray(self.aug.augment_image(np.array(image)).copy())

        if self.transform:
            image = self.transform(image)

        return image, label
        
    def __len__(self):
        return len(self.images)
