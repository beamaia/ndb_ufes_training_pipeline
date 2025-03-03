from abc import ABC
import os
import pathlib as pl
from datetime import datetime
import random

import mlflow
import numpy as np
import torch
import tqdm

from .job import BaseJob, JobType

class TestJob(BaseJob):
    model = None
    loss_func = None

    def _load_model(self, path):
        # model class initialized
        model = None

        if not os.path.exists(path):
            raise FileNotFoundError(f"File to path {path} not found.")

        torch.load()
        return model

    def _create_artifacts(self, pred, labels, images):
        num_classes = set(labels)

        # n x 5 grid image of wrong predictions
        wrong_labels = pred != labels
        wrong_ids = [i for i, wrong_pred in enumerate(wrong_labels) if wrong_pred]

        wrong_classes_ids = {class_id: [i for i in wrong_ids if pred[i] == class_id] for class_id in num_classes}
        wrong_classes_images = {class_id: [images[i] for i in list_ids] for class_id, list_ids in wrong_classes_ids.items()}
        wrong_random_images = {class_id: random.sample(images_subset, 5) for class_id, images_subset in wrong_classes_images.items()}

        # n x 5 grid image of correct predictions
        correct_labels = pred == labels
        correct_ids = [i for i, correct_pred in enumerate(correct_labels) if correct_pred]

        correct_classes_ids = {class_id: [i for i in correct_ids if pred[i] == class_id] for class_id in num_classes}
        correct_classes_images = {class_id: [images[i] for i in list_ids] for class_id, list_ids in correct_classes_ids.items()}
        correct_random_images = {class_id: random.sample(images_subset, 5) for class_id, images_subset in correct_classes_images.items()}


    def _test(self, test_dataloader, create_artifacts=True):
        artifacts = None
        self.model.eval()

        with torch.no_grad():
            running_loss = 0
            pred_list, labels_list = [], []

            for i, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                if self.loss_func:
                    loss = self.loss_func(outputs, labels)
                    running_loss += loss.item()

                _, pred = torch.max(outputs, axis=1)
                pred_list.extend(pred.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        test_acc = np.mean(np.array(pred_list) == np.array(labels_list))
        test_loss = running_loss / (i + 1) if self.loss_func else None

        # if create_artifact:
        #   artifacts = self._create_artifacts(pred, labels, images)

        return test_acc, test_loss, artifacts
    
    def run (self, test_dataloader, model=None, loss_func=None, create_artifacts=True):
        self.model = model if model else self._load_model()
        self.loss_func = loss_func

        test_acc, test_loss, artifacts = self._test(test_dataloader, create_artifacts)

        if not create_artifacts:
            return test_acc, test_loss
        return test_acc, test_loss, artifacts