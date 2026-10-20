import warnings
from datetime import datetime
from enum import Enum
from typing import Literal

from aenum import MultiValueEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    """A misspelled or removed key in params.yaml is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ProjectEnum(str, Enum):
    multiclass = "multiclass"

    def __str__(self):
        return self.value


class LossEnum(str, MultiValueEnum):
    cross_entropy = "cross_entropy", "CrossEntropyLoss", "CrossEntropy"


class OptimizerEnum(str, MultiValueEnum):
    sgd = "sgd", "SGD"
    adam = "adam", "Adam"


class SchedulerEnum(str, MultiValueEnum):
    reduce_lr_on_plateau = "reduce_lr_on_plateau", "ReduceLROnPlateau"
    step_r = "step_lr", "StepLR", "steplr"


class ModelEnum(str, Enum):
    resnet50 = "resnet50"
    mobilenetv2 = "mobilenetv2"
    densenet121 = "densenet121"
    vgg16 = "vgg16"
    efficientnetb4 = "efficientnetb4"
    efficientnet_b0 = "efficientnet_b0"
    efficientnet_b3 = "efficientnet_b3"
    convnext_large = "convnext_large"
    swin_tiny_patch4_window7_224 = "swin_tiny_patch4_window7_224"
    swin_base_patch4_window7_224 = "swin_base_patch4_window7_224"
    coat_lite_small = "coat_lite_small"
    pit_s_distilled_224 = "pit_s_distilled_224"
    vit_small_patch16_384 = "vit_small_patch16_384"


MODEL_FAMILY = {
    **dict.fromkeys(["resnet50", "mobilenetv2", "densenet121", "vgg16", "efficientnetb4", "efficientnet_b0", "efficientnet_b3", "convnext_large"], "cnn"),
    **dict.fromkeys(["coat_lite_small", "pit_s_distilled_224", "vit_small_patch16_384", "swin_tiny_patch4_window7_224", "swin_base_patch4_window7_224"], "transformer"),
}


class Optimizer(Strict):
    name: OptimizerEnum = OptimizerEnum.sgd
    learning_rate: float = 0.001
    momentum: float = 0.9
    other: dict | None = None


class Scheduler(Strict):
    name: SchedulerEnum = SchedulerEnum.reduce_lr_on_plateau
    other: dict | None = None


class Model(Strict):
    name: ModelEnum
    other: dict | None = None


class OtherHyperparams(Strict):
    loss: LossEnum
    loss_weights: bool = True
    epochs: int = Field(200, ge=1)
    batch_size: int = Field(30, ge=2)


class Hyperparameters(Strict):
    optimizer: Optimizer
    scheduler: Scheduler
    model: Model
    other: OtherHyperparams


class Dataset(Strict):
    patch_dir: str  # the folder of the patch images, <patch_id>.png
    folds_csv: str  # fold_assignments.csv made by the data organizer
    patch_table: str | None = None  # patch_table_public.csv of the data organizer; adds the picture of each patch, for the per-picture metrics
    folds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    test_fold: int = 5  # kept out of the cross-validation; every fold model is tested on it

    @model_validator(mode="after")
    def validate_folds(self):
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("dataset.folds contains duplicate folds")
        if self.test_fold not in self.folds:
            raise ValueError("dataset.test_fold must be one of dataset.folds")
        if len(self.folds) < 3:
            raise ValueError("dataset.folds needs at least three folds: train, validation and test")
        return self


class RunSettings(Strict):
    purpose: str = "final"  # names the MLflow experiment, ndb_ufes_<purpose>; trials and tests use another name
    suffix: str = ""  # added to the run names
    hourly_cost: float | None = None  # price of the machine per hour in dollars; the cost of the run is logged
    tracking_uri: str | None = None  # MLflow server; the credentials stay in .env
    last_checkpoints: int = Field(1, ge=0)  # besides the best checkpoint, each fold keeps the checkpoints of its last epochs, this many (1 is the last epoch); the older ones are deleted
    checkpoints_in_mlflow: bool = False  # a copy of each checkpoint in the MLflow artifacts; off by default, it doubles the disk space


class Training(Strict):
    mode: Literal["finetune", "frozen_head"] = "finetune"


class EarlyStopping(Strict):
    enabled: bool = True
    patience: int = Field(10, ge=1)
    min_delta: float = Field(0.001, ge=0)
    mode: Literal["all", "local"] = "all"


class Experiment(Strict):
    models: list[ModelEnum] = Field(default_factory=list)


class Parameters(Strict):
    project: ProjectEnum
    dataset: Dataset
    hyperparameters: Hyperparameters
    training: Training = Field(default_factory=Training)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)
    experiment: Experiment = Field(default_factory=Experiment)
    run: RunSettings = Field(default_factory=RunSettings)
    device: str = "cpu"
    node_type: str = "float32"
    seed: int = Field(42, ge=0)
    resume: bool = False
    run_name: str

    @model_validator(mode="after")
    def run_name_not_empty(self):
        if not self.run_name:
            self.run_name = f"{self.hyperparameters.model.name.value}_{self.hyperparameters.optimizer.name.value}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            warnings.warn(f"Empty run name, will be set {self.run_name}", UserWarning)
        return self

    def __str__(self):
        hyper, other = self.hyperparameters, self.hyperparameters.other
        return "\n".join([
            "-" * 90,
            f"* Project: {self.project.value} - Run: {self.run_name}",
            f"* Patch images: {self.dataset.patch_dir}",
            f"* Folds file: {self.dataset.folds_csv}; folds {self.dataset.folds}; test fold {self.dataset.test_fold}; patch table {self.dataset.patch_table}",
            f"* Model: {hyper.model.name.value} ({self.training.mode}); device {self.device}; seed {self.seed}",
            f"* Optimizer: {hyper.optimizer.name.value}, learning rate {hyper.optimizer.learning_rate}, momentum {hyper.optimizer.momentum}",
            f"* Scheduler: {hyper.scheduler.name.value} {hyper.scheduler.other or ''}",
            f"* Loss: {other.loss} (weighted {other.loss_weights}); epochs {other.epochs}; batch size {other.batch_size}",
            f"* Early stopping: {self.early_stopping.enabled}, patience {self.early_stopping.patience}, min delta {self.early_stopping.min_delta}, mode {self.early_stopping.mode}",
            "-" * 90,
        ])
