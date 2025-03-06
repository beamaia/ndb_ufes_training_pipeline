from abc import ABC
import os
import pathlib as pl
from datetime import datetime
import random

import mlflow
import numpy as np
import torch
import tqdm

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

from .job import BaseJob, JobType
from src import logger
from src.models.model_selector import ModelSelector

class TestJob(BaseJob):
    model = None
    loss_func = None

    def _prepare_model(self, path):
        model_str = self.params.hyperparameters.model.name.value
        model_obj =  ModelSelector(model_str, num_classes=self.num_classes).model.to(self.device)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File to path {path} not found.")
        
        print("models state dict", model_obj_keys.state_dict())
        model_obj_keys = [key for key, _ in model_obj_keys.state_dict().item()]
        
        check_model_dict = torch.load(path, weights_only=True)
        check_model_keys = [key for key, _ in check_model_dict.item()]

        if all([ True if key in model_obj_keys else False for key in check_model_keys]):
            model_obj.load_state_dict(check_model_dict)
        else:
            print(check_model_keys)
            print(model_obj_keys)

        try:
            model_obj.load_state_dict(torch.load(path, weights_only=True))
        except Exception as e:
            logger.error("Exception caught while trying to load model.")
            logger.error(e.__str__)
            logger.error(e.args)
            logger.error(e.__traceback__)
            quit()

        return model_obj
    
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


    def _test(self, test_dataloader, create_artifacts=True, stage="val"):
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
        test_balanced_acc = balanced_accuracy_score(labels_list, pred_list)
        test_recall = recall_score(labels_list, pred_list)
        test_precision = precision_score(labels_list, pred_list)

        metrics =  {
            f"{stage}_acc": test_acc
        }

        if self.loss_func:
            metrics[f"{stage}_loss"] =  test_loss

        metrics[f"{stage}_balanced_acc"] =  test_balanced_acc
        metrics[f"{stage}_recall"] =  test_recall
        metrics[f"{stage}_precision"] =  test_precision

        # if create_artifact:
        #   artifacts = self._create_artifacts(pred, labels, images)

        return metrics, artifacts
    
    def run (self, test_dataloader, model=None, loss_func=None, create_artifacts=True, path=None, stage="val"):
        self.model = model
        self.loss_func = loss_func

        if not self.params.stages.train:
            print("Not implemented yet")
        elif stage == "test":
            self.model = self._prepare_model(path)
        
        metrics, artifacts = self._test(test_dataloader, create_artifacts, stage)

        if not create_artifacts:
            return metrics
        return metrics, artifacts