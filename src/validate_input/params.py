import pathlib as pl
import warnings

from enum import Enum
from aenum import MultiValueEnum
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

class ProjectEnum(str, Enum):
    oscc_bin = "oscc_bin"
    dys_bin = "dys_bin"
    oscc_dys = "oscc_dys"
    multiclass = "multiclass"

class LossEnum(str, MultiValueEnum):
    cross_entropy = "cross_entropy", "CrossEntropyLoss", "CrossEntropy"

class OptimizerEnum(str, MultiValueEnum):
    sgd = "sgd" , "SGD"
    adam = "Adam", "adam"

class SchedulerEnum(str, MultiValueEnum):
    reduce_lr_on_plateau = "reduce_lr_on_plateau", "ReduceLROnPlateau"
    step_r = "step_lr", "StepLR", "steplr"

class TrainTypeEnum(str, MultiValueEnum):
    holdout = "holdout", "hold_out", "Holdout"
    cross_validation = "cross_validation", "CrossValidation"

class Test(BaseModel):
    model_path: str = Field(None, alias="model_path")

class ModelEnum(str, Enum):
    resnet50 = "resnet50"
    mobilenet = "mobilenet"
    densenet121 = "densenet121"
    vgg16 = "vgg16"
    coat_lite_small = "coat_lite_small"
    pit_s_distilled_224 = "pit_s_distilled_224"
    vit_small_patch_384 = "vit_small_patch16_384"

class Optimizer(BaseModel):
    name: OptimizerEnum = Field(OptimizerEnum.sgd, alias="name")
    learning_rate: float = Field(0.001, alias="learning_rate")
    momentum: float = Field(0.9, alias="momentum")
    other: dict | None = Field(None, alias="other")

class Scheduler(BaseModel):
    name: SchedulerEnum = Field(SchedulerEnum.reduce_lr_on_plateau, alias="name")
    other: dict | None = Field(None, alias="other")

class Model(BaseModel):
    name: ModelEnum = Field(alias="name")
    other: dict | None = Field(None, alias="other")

class OtherHyperparams(BaseModel):
    loss: LossEnum = Field(alias="loss")
    loss_weights: bool  = Field(True, alias="loss_weights")
    folds: int = Field(5, alias="folds", min=1)
    epochs: int = Field(200, alias="epochs", min=1)
    batch_size: int = Field(30, alias="batch_size", min=2)
    train_type: TrainTypeEnum | None = Field(TrainTypeEnum.cross_validation, alias="train_type")

class Hyperparameters(BaseModel):
    optimizer: Optimizer = Field(alias="optimizer")
    scheduler: Scheduler = Field(alias="scheduler")
    model: Model = Field(alias="model")
    other: OtherHyperparams = Field(alias="other")

class Dataset(BaseModel):
    root: str = Field("data", alias="data")
    patch: str = Field("patch_images", alias="patch_images")
    origin: str = Field("original_images", alias="original_images")

class Stages(BaseModel):
    train: bool = Field(True, alias="train")
    test: bool = Field(True, alias="test")
    origin_test: bool = Field(False, alias="origin_test")
    
class Parameters(BaseModel):
    project: ProjectEnum = Field(alias="project")
    stages: Stages = Field(alias="stages")
    dataset: Dataset = Field(alias="dataset")
    hyperparameters: Hyperparameters = Field(alias="hyperparameters")
    test: Test | None = Field(None, alias="origin_test")
    device: str = Field("mps", alias="device")
    node_type: str = Field("bfloat16", alias="node_type")
    run_name: str = Field(alias="run_name")

    @model_validator(mode="after")
    def run_name_not_empty(self):
        run_name = self.run_name
        model_name = self.hyperparameters.model.name.value
        optimizer_name = self.hyperparameters.optimizer.name.value
        if not len(run_name):
            new_run_name  = model_name + "_" +\
                   optimizer_name + "_" +\
                   datetime.now().strftime("%Y_%m_%d_%H_%M_%S%Z")
            
            warnings.warn(f"Empty run name, will be set {new_run_name}", UserWarning)
            self.run_name  = new_run_name
        return self
    
    def __str__(self):
        dashes = "-"*90
        message = f"""
{dashes}
* Project: {self.project.value} - Run: {self.run_name}
{dashes}
* Stages to execute:
"""
        stages_to_execute = "\n".join(["- "+ stage[0] for stage in self.stages if stage[1]])
        message = message + stages_to_execute

        root_path = pl.Path(self.dataset.root)
        patch_path = pl.Path(root_path / self.dataset.patch)
        origin_path = pl.Path(root_path / self.dataset.origin)

        message = message + f"""
{dashes} 
* Dataset path:
patch: {patch_path}
origin: {origin_path}
{dashes}
* Hyperparams:
{dashes}
** Model:
name: {self.hyperparameters.model.name.value}
"""        
        other_model = self.hyperparameters.model.other
        if other_model != None or (isinstance(other_model, dict) and other_model):
            other_model_str = "\n".join([f"{key}: {item}" for key, item in other_model.items()])
            message = message + other_model_str

        message = message + f"""{dashes}
** Optimizer:
name: {self.hyperparameters.optimizer.name.value}
learning_rate: {self.hyperparameters.optimizer.learning_rate}
{dashes}
** Scheduler:
name: {self.hyperparameters.scheduler.name.value}
"""
        other_scheduler = self.hyperparameters.scheduler.other
        if other_scheduler != None or (isinstance(other_scheduler, dict) and other_scheduler):
            other_scheduler_str = "\n".join([f"{key}: {item}" for key, item in other_scheduler.items()])
            message = message + other_scheduler_str

        message = message + f"""
{dashes}
** Other:
loss: {self.hyperparameters.other.loss}
weighted_loss: {self.hyperparameters.other.loss_weights}
folds: {self.hyperparameters.other.folds}
epochs: {self.hyperparameters.other.epochs}
batch_size: {self.hyperparameters.other.batch_size}"""
        
        if self.test and self.test.model_path:
            message = message + f"""
{dashes}
* Origin test info:
model_path: {self.test.model_path}
"""
        return message