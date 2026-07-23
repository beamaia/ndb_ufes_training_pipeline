from abc import ABC
import os
import pathlib as pl
import random
from datetime import datetime
from tqdm import tqdm

import mlflow
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .job import BaseJob, JobType
from .test import TestJob

from src import logger
from src.models.model_selector import ModelSelector
from src.optimizer.optimizer_scheduler_selector import OptimizerSchedulerSelector
from src.loss.loss_selector import LossSelector
from src.early_stopper import EarlyStopper
from src.pipeline.metrics import classification_metrics

class TrainJob(BaseJob):
    job: JobType = JobType.train

    model = None
    optimizer = None
    scheduler = None
    loss_func = None

    def __init__(self, params, num_classes):
        self.params = params
        self.num_classes = num_classes
        self.device = self.params.device 
        self.checkpoint_base = pl.Path("models")
        self.checkpoint_root = self.checkpoint_base
        self.models_saved = set()
        self.resume = self.params.resume

        self._test = TestJob(params)
        self.node_type = getattr(torch, self.params.node_type)


    def _prepare_model(self):
        model_str = self.params.hyperparameters.model.name.value
        return ModelSelector(
            model_str,
            num_classes=self.num_classes,
            training_mode=self.params.training.mode,
        ).model.to(self.device).to(self.node_type)
    
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
                                              model_params=(p for p in self.model.parameters() if p.requires_grad),
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
            weights = torch.tensor(weights, dtype=self.node_type, device=self.device)
        
        return LossSelector(loss_name, use_weight, weights=weights).loss.to(self.device).to(self.node_type)

    def _inner_train_loop(self, train_dataloader):
        self.model.train()
        running_loss = 0

        pred_list, labels_list = [], []
        if len(train_dataloader.dataset) == 0:
            raise ValueError("Training split is empty.")
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

        train_loss = running_loss / (i+1)
        metrics = classification_metrics(
            labels_list, pred_list, self.num_classes, prefix="train"
        )
        metrics["train_loss"] = float(train_loss)
        return metrics

    def _create_artifacts_dir(self):
        artifact_root = self.checkpoint_base / f"project_{self.params.project}_run_{self.params.run_name}"

        artifact_root.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_root = artifact_root

    def _training_state_path(self, checkpoint_name):
        return pl.Path(str(checkpoint_name) + "_training_state.pt")

    def _save_training_state(
        self,
        state_path,
        epoch_completed,
        best_loss,
        best_loss_metrics,
        last_metrics,
        early_stopper,
        train_dataloader,
        completed=False,
    ):
        generator = getattr(train_dataloader, "generator", None)
        state = {
            "epoch_completed": epoch_completed,
            "completed": completed,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": best_loss,
            "best_loss_metrics": best_loss_metrics,
            "last_metrics": last_metrics,
            "early_stopper": None if early_stopper is None else {
                "counter": early_stopper.counter,
                "min_validation_loss": early_stopper.min_validation_loss,
            },
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "train_generator_state": generator.get_state() if generator is not None else None,
        }
        temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(state, temporary_path)
        temporary_path.replace(state_path)

    def _restore_training_state(self, state_path, early_stopper, train_dataloader):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        if self.scheduler and state["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if early_stopper and state["early_stopper"] is not None:
            early_stopper.counter = state["early_stopper"]["counter"]
            early_stopper.min_validation_loss = state["early_stopper"]["min_validation_loss"]
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        generator = getattr(train_dataloader, "generator", None)
        if generator is not None and state["train_generator_state"] is not None:
            generator.set_state(state["train_generator_state"])
        return state

    def _train(self, train_dataloader, val_dataloader, fold=0):
        fold_path = self.checkpoint_root / f"fold_{fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        checkpoint_name = fold_path / (
            f"model_{self.params.hyperparameters.model.name.value}_"
            f"optim_{self.params.hyperparameters.optimizer.name.value}"
        )
        state_path = self._training_state_path(checkpoint_name)
        best_model_path = pl.Path(str(checkpoint_name) + "_best.pt")

        logger.info(f"Start training at {datetime.now()}")

        early_stopping = self.params.early_stopping
        early_stopper = None
        if early_stopping.enabled:
            early_stopper = EarlyStopper(
                patience=early_stopping.patience,
                min_delta=early_stopping.min_delta,
                mode=early_stopping.mode,
            )
            mlflow.log_params({
                "early_stopping_enabled": early_stopping.enabled,
                "early_stopping_patience": early_stopping.patience,
                "early_stopping_min_delta": early_stopping.min_delta,
                "early_stopping_mode": early_stopping.mode,
            })

        num_epochs = self.params.hyperparameters.other.epochs
        best_loss = np.inf
        best_loss_metrics = {}
        start_epoch = 0

        if self.resume and state_path.exists():
            state = self._restore_training_state(state_path, early_stopper, train_dataloader)
            start_epoch = state["epoch_completed"]
            best_loss = state["best_loss"]
            best_loss_metrics = state["best_loss_metrics"]
            if best_model_path.exists():
                self.models_saved.add(str(best_model_path))
            if state["completed"]:
                logger.info(f"Fold {fold} is already complete; reusing its saved training state.")
                return state["last_metrics"], best_loss_metrics
            logger.info(f"Resuming fold {fold} from epoch {start_epoch + 1}.")
            mlflow.log_param("resumed_from_epoch", start_epoch)

        checkpoint_epochs = set([int(num_epochs * per / 100) for per in np.arange(0, 100, 10)])

        last_metrics = {}
        for epoch in tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs):
            train_metrics = self._inner_train_loop(train_dataloader)
            val_metrics, _ = self._test.run(val_dataloader, 
                                            model=self.model, 
                                            loss_func=self.loss_func, 
                                            create_artifacts=False)
            
            metrics = train_metrics | val_metrics

            logger.info(f"Epoch: {epoch + 1} | Train accuracy: {metrics['train_acc'] * 100: 0.3f}% | Train loss: {metrics['train_loss']:0.3f} | Validation accuracy: {metrics['val_acc'] * 100: 0.3f}% | Validation loss: {metrics['val_loss']: 0.3f}")
            logger.info(f"Epoch: {epoch + 1} | Other train metrics | Balanced accuracy: {metrics['train_balanced_acc'] * 100: 0.3f}% | Macro F1: {metrics['train_macro_f1'] * 100: 0.3f}% | Recall: {metrics['train_recall'] * 100: 0.3f}% | Precision {metrics['train_precision'] * 100: 0.3f}%")
            logger.info(f"Epoch: {epoch + 1} | Other validation metrics | Balanced accuracy: {metrics['val_balanced_acc'] * 100: 0.3f}% | Macro F1: {metrics['val_macro_f1'] * 100: 0.3f}% | Recall: {metrics['val_recall'] * 100: 0.3f}% | Precision {metrics['val_precision'] * 100: 0.3f}%")
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

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics["val_loss"])
            else:
                self.scheduler.step()
            last_metrics = metrics
            should_stop = early_stopper and early_stopper.early_stop(metrics["val_loss"])
            self._save_training_state(
                state_path,
                epoch + 1,
                best_loss,
                best_loss_metrics,
                last_metrics,
                early_stopper,
                train_dataloader,
            )
            if should_stop:
                mlflow.log_metric("early_stopped_epoch", epoch + 1)
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"after {early_stopping.patience} stale validation-loss epochs."
                )
                break

        completed_epoch = epoch + 1 if last_metrics else start_epoch
        self._save_training_state(
            state_path,
            completed_epoch,
            best_loss,
            best_loss_metrics,
            last_metrics,
            early_stopper,
            train_dataloader,
            completed=True,
        )
        
        logger.info(f"Finished training at {datetime.now()}")
        
        return last_metrics, best_loss_metrics

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
