import os

import numpy as np
import pandas as pd
import torch

from .job import BaseJob
from src import logger
from src.checkpoint import load_checkpoint
from src.pipeline.metrics import class_key, classification_metrics, confusion_metrics, roc_auc_metrics


class TestJob(BaseJob):
    """Evaluates a model on a dataloader: a model loaded from a checkpoint, or the model that is being trained."""

    model = None
    loss_func = None

    def __init__(self, params, num_classes=None, model_path=None):
        self.params = params
        self.device = self.params.device
        self.num_classes = num_classes
        self.node_type = getattr(torch, self.params.node_type)
        self.model = self._prepare_model(model_path) if model_path and self.num_classes else None

    def _prepare_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File to path {path} not found.")
        try:
            model, description = load_checkpoint(path, self.device, self.node_type)
        except Exception as exc:
            logger.error(str(exc))
            raise RuntimeError(f"Could not load checkpoint: {path}") from exc
        logger.info(f"Loaded {description['model_name']} (validation fold {description['fold']}, epoch {description['epoch']}) from {path}")
        return model

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
            true_pred_df[f"prob_{class_key(class_name)}"] = [float(row[class_index]) for row in probability_list]
        return true_pred_df

    def run(self, dataloader, model=None, loss_func=None, stage="val"):
        """The metrics and a table of the predictions (with the class probabilities) on the dataloader."""
        if model is not None:
            self.model = model
        if len(dataloader.dataset) == 0:
            raise ValueError(f"{stage} split is empty.")

        loss_func = loss_func.to(self.device) if loss_func else None
        self.model.eval()

        with torch.no_grad():
            running_loss = 0
            pred_list, labels_list, probability_list = [], [], []

            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                if loss_func:
                    running_loss += loss_func(outputs, labels).item()

                _, pred = torch.max(outputs, axis=1)
                pred_list.extend(pred.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probability_list.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        class_names = [name for name, _ in sorted(dataloader.dataset.classes.items(), key=lambda item: item[1])]
        metrics = classification_metrics(labels_list, pred_list, len(class_names), prefix=stage)
        metrics |= roc_auc_metrics(labels_list, np.array(probability_list), class_names, prefix=stage)
        metrics |= confusion_metrics(labels_list, pred_list, class_names, prefix=stage)
        if loss_func:
            metrics[f"{stage}_loss"] = running_loss / (i + 1)
        return metrics, self._prediction_table(dataloader.dataset, labels_list, pred_list, probability_list, stage)
