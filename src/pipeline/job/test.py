import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

from .job import BaseJob, JobType
from src import logger
from src.models.model_selector import ModelSelector

class TestJob(BaseJob):
    model = None
    loss_func = None

    def __init__(self, params, num_classes=None, model_path=None):
        self.params = params
        self.device = self.params.device 
        self.num_classes = num_classes
        self.node_type = getattr(torch, self.params.node_type)
        self.model = self._prepare_model(model_path) if model_path and self.num_classes else None

    def _prepare_model(self, path):
        model_str = self.params.hyperparameters.model.name.value
        model_obj = ModelSelector(
            model_str,
            num_classes=self.num_classes,
            training_mode=self.params.training.mode,
        ).model.to(self.device).to(self.node_type)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File to path {path} not found.")
        
        model_obj_keys = [key for key, _ in model_obj.state_dict().items()]
        
        check_model_dict = torch.load(path, weights_only=True)
        check_model_keys = [key for key, _ in check_model_dict.items()]

        
        try:
            if all([ True if key in model_obj_keys else False for key in check_model_keys]):
                model_obj.load_state_dict(check_model_dict)
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
        model_name = self.params.hyperparameters.model.name.value
        if model_name == "vgg16":
            device = "cpu"
            self.model = self.model.cpu()
        else:
            device = self.device

        self.model.eval()

        with torch.no_grad():
            running_loss = 0
            pred_list, labels_list, probability_list = [], [], []

            for i, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)

                if self.loss_func:
                    loss = self.loss_func(outputs, labels)
                    running_loss += loss.item()

                _, pred = torch.max(outputs, axis=1)
                probabilities = torch.softmax(outputs, dim=1)
                pred_list.extend(pred.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probability_list.extend(probabilities.cpu().numpy())

        test_acc = np.mean(np.array(pred_list) == np.array(labels_list))
        test_loss = running_loss / (i + 1) if self.loss_func else None
        test_balanced_acc = balanced_accuracy_score(labels_list, pred_list)
        test_recall = recall_score(labels_list, pred_list, average='micro' if len(np.unique(labels_list) > 2) else None)
        test_precision = precision_score(labels_list, pred_list, average='micro' if len(np.unique(labels_list) > 2) else None)

        metrics =  {
            f"{stage}_acc": test_acc
        }

        if self.loss_func:
            metrics[f"{stage}_loss"] =  test_loss

        metrics[f"{stage}_balanced_acc"] =  test_balanced_acc
        metrics[f"{stage}_recall"] =  test_recall
        metrics[f"{stage}_precision"] =  test_precision

        true_pred_df = self._prediction_table(
            test_dataloader.dataset,
            labels_list,
            pred_list,
            probability_list,
            stage,
        )
        # if create_artifact:
        #   artifacts = self._create_artifacts(pred, labels, images)

        self.model.to(self.device)

        return metrics, true_pred_df, artifacts

    def _prediction_table(self, dataset, labels_list, pred_list, probability_list, stage):
        metadata = getattr(dataset, "metadata", [])
        true_pred_df = pd.DataFrame(metadata) if len(metadata) == len(labels_list) else pd.DataFrame()
        true_pred_df = true_pred_df.assign(**{
            "y_true": labels_list,
            "y_pred": pred_list,
            "stage": stage,
            "model": self.params.hyperparameters.model.name.value,
        })
        class_names = [name for name, _ in sorted(dataset.classes.items(), key=lambda item: item[1])]
        for class_index, class_name in enumerate(class_names):
            safe_name = class_name.lower().replace(" ", "_").replace("-", "_")
            true_pred_df[f"prob_{safe_name}"] = [float(row[class_index]) for row in probability_list]
        return true_pred_df
    
    def run (self, test_dataloader, model=None, loss_func=None, create_artifacts=True, path=None, stage="val"):
        self.loss_func = loss_func

        if self.params.stages.train and model:
            self.model = model
        
        metrics, true_pred_df, artifacts = self._test(test_dataloader, create_artifacts, stage)

        if not create_artifacts:
            return metrics, true_pred_df
        return metrics, true_pred_df, artifacts
