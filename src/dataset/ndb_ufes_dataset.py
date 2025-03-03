import os
from cv2 import imread, IMREAD_COLOR
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class NDBUfesDataset(Dataset):
    def __init__(self, images, labels, classes_dict, transform = None):
        self.images = images
        self.labels = labels
        self.classes = classes_dict
        self.transform = transform

    def __getitem__(self, index):
        if index >= len(self.images):
            raise IndexError("Index out of bounds in target dataset.")
        
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError(f"File not found in path {self.images[index]}")
        
        image = imread(self.images[index], IMREAD_COLOR)
        label = self.labels[index]

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label
        
    def __len__(self):
        return 0 if not isinstance(self.images, np.ndarray) else len(self.images)
