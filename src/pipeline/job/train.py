from abc import ABC
import os
import pathlib as pl
from datetime import datetime
from tqdm import tqdm

import mlflow
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

from .job import BaseJob, JobType
from .test import TestJob

from src import logger
from src.models.model_selector import ModelSelector
from src.optimizer.optimizer_scheduler_selector import OptimizerSchedulerSelector
from src.loss.loss_selector import LossSelector
from src.early_stopper import EarlyStopper

class TrainJob(BaseJob):
    job: JobType = JobType.train

    model = None
    optimizer = None
    scheduler = None
    loss_func = None

    checkpoint_root = pl.Path("models")
    models_saved = set()

    def __init__(self, params, num_classes):
        self.params = params
        self.num_classes = num_classes
        self.device = self.params.device 

        self._test = TestJob(params)
        self.node_type = getattr(torch, self.params.node_type)


    def _prepare_model(self):
        model_str = self.params.hyperparameters.model.name.value
        return ModelSelector(model_str, num_classes=self.num_classes).model.to(self.device).to(self.node_type)
    
    def _prepare_optim_sched(self):
        optim = self.params.hyperparameters.optimizer
        optim_str = optim.name.value
        learning_rate = optim.learning_rate
        momentum = optim.momentum
        optim_params = optim.other

        sched = self.params.hyperparameters.scheduler
        sched_str = sched.name.value
        sched_params = sched.other

        selector = OptimizerSchedulerSelector(optimizer=optim_str,
                                              learning_rate=learning_rate,
                                              momentum=momentum,
                                              model_params=self.model.parameters(),
                                              scheduler=sched_str,
                                              optimizer_params=optim_params,
                                              scheduler_params=sched_params)
        
        return selector.optimizer, selector.scheduler
    
    def _prepare_loss_func(self, dataset):
        loss_name = self.params.hyperparameters.other.loss.value
        use_weight = self.params.hyperparameters.other.loss_weights
        weights = None

        if loss_name == "cross_entropy" and use_weight:
            weights = compute_class_weight('balanced', classes=np.unique(dataset.labels), y=dataset.labels)
        
        return LossSelector(loss_name, weights, weights=None).loss.to(self.device).to(self.node_type)

    def _inner_train_loop(self, train_dataloader):
        self.model.train()
        running_loss = 0

        pred_list, labels_list = [], []
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)

            loss = self.loss_func(outputs, labels)
            running_loss += loss.item()
            
            _, pred = torch.max(outputs, axis=1)
            pred_list.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_acc = np.mean(np.array(pred_list) == np.array(labels_list))
        train_loss = running_loss / (i+1)
        train_balanced_acc = balanced_accuracy_score(labels_list, pred_list)
        train_recall = recall_score(labels_list, pred_list, average='micro' if len(np.unique(labels_list) > 2) else None)
        train_precision = precision_score(labels_list, pred_list, average='micro' if len(np.unique(labels_list) > 2) else None)

        return dict(
            train_acc=train_acc,
            train_loss=train_loss,
            train_balanced_acc=train_balanced_acc,
            train_recall=train_recall,
            train_precision=train_precision
        )

    def _create_artifacts_dir(self):
        artifact_root = self.checkpoint_root  / f"project_{self.params.project}_run_{self.params.run_name}"

        if not os.path.exists(artifact_root):
            os.mkdir(artifact_root)
        
        self.checkpoint_root = artifact_root

    def _train(self, train_dataloader, val_dataloader, fold=0):
        if not fold:
            checkpoint_name = self.checkpoint_root / f"model_{self.params.hyperparameters.model.name.value}_optim_{self.params.hyperparameters.optimizer.name.value}"
        else:
            fold_path = self.checkpoint_root / f"fold_{fold}"
            os.mkdir(fold_path)
            checkpoint_name = fold_path / f"model_{self.params.hyperparameters.model.name.value}_optim_{self.params.hyperparameters.optimizer.name.value}"

        logger.info(f"Start training at {datetime.now()}")

        early_stopper = EarlyStopper()

        num_epochs = self.params.hyperparameters.other.epochs
        best_loss = np.inf
        best_loss_metrics = {}

        checkpoint_epochs = set([int(num_epochs * per / 100) for per in np.arange(0, 100, 10)])

        for epoch in tqdm(range(num_epochs)):
            train_metrics = self._inner_train_loop(train_dataloader)
            val_metrics, _ = self._test.run(val_dataloader, 
                                            model=self.model, 
                                            loss_func=self.loss_func, 
                                            create_artifacts=False)
            
            metrics = train_metrics | val_metrics

            logger.info(f"Epoch: {epoch + 1} | Train accuracy: {metrics["train_acc"] * 100: 0.3f}% | Train loss: {metrics["train_loss"]:0.3f} | Validation accuracy: {metrics["val_acc"] * 100: 0.3f}% | Validation loss: {metrics["val_loss"]: 0.3f}")
            logger.info(f"Epoch: {epoch + 1} | Other train metrics | Balanced accuracy: {metrics["train_balanced_acc"] * 100: 0.3f}% | Recall: {metrics["train_recall"] * 100: 0.3f}% | Precision {metrics["train_precision"] * 100: 0.3f}%")
            logger.info(f"Epoch: {epoch + 1} | Other validation metrics | Balanced accuracy: {metrics["val_balanced_acc"] * 100: 0.3f}% | Recall: {metrics["val_recall"] * 100: 0.3f}% | Precision {metrics["val_precision"] * 100: 0.3f}%")
            logger.info(f"Epoch: {epoch + 1} | Learning rate: {self.optimizer.param_groups[0]['lr']}")

            if epoch in checkpoint_epochs:
                torch.save(self.model.state_dict(), str(checkpoint_name) + "_checkpoint.pt")
                self.models_saved.add(str(checkpoint_name) + "_checkpoint.pt")

            if best_loss > metrics["val_loss"]:
                best_loss = metrics["val_loss"]
                torch.save(self.model.state_dict(), str(checkpoint_name) + "_best.pt")
                self.models_saved.add(str(checkpoint_name) + "_best.pt")
                best_loss_metrics = {key:item for key, item in metrics.items() if "val" in key}
                
            mlflow.log_metrics(metrics, step=epoch)
            mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=epoch)

            self.scheduler.step(metrics["val_loss"])
            if early_stopper.early_stop(metrics["val_loss"]):
                break
        
        logger.info(f"Finished training at {datetime.now()}")
        
        return metrics, best_loss_metrics

    def run(self, train_dataloader, val_dataloader, fold=0):
        if not os.path.exists(self.checkpoint_root):
            os.mkdir(self.checkpoint_root)

        # torch.set_default_device(self.params.device)

        self.model = self._prepare_model()
        self.optimizer, self.scheduler = self._prepare_optim_sched()

        self.loss_func = self._prepare_loss_func(train_dataloader.dataset)
        
        self._create_artifacts_dir()
        metrics, best_loss_metrics = self._train(train_dataloader, val_dataloader, fold)
        
        return metrics, best_loss_metrics

