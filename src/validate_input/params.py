from pydantic import BaseModel, Field, field_validator
import warnings
# project: # expected options [oscc_notoscc, dys_notdys, oscc_dys, multiclass]
# run_name: ""

# stages:
#  - train: true
#  - test: true

# dataset:
#  - root: data
#  - patch: patch_images
#  - origin: original_images

# hyperparameters:
#  - optimizer:
#    - name: sgd
#    - learning_rate: 0.001
#  - scheduler:
#    - name: reduce_lr_on_plateau
#    - other:
#       - factor: 0.1
#       - patience: 5
#       - min_lr: 0.000001
#  - model:
#    - name:
#    - other:
#  - other:
#    - loss: cross_entropy
#    - loss_weights: true
#    - folds: 5
#    - epochs: 200
#    - batch_size: 30

class Hyperparameters(BaseModel):
    pass

class Dataset(BaseModel):
    pass

class Stages(BaseModel):
    pass

class Parameters(BaseModel):
    project: str = Field(None, )
    run_name: str = Field()
    stages: Stages = Field(None, "stages", )
    dataset: Dataset = Field(None, "dataset")
    hyperparameters: Hyperparameters = Field(None, "hyperparameters")

    @field_validator()
    def run_name_not_empty(cls, v):
        if not len(v):
            warnings.warn(f"Empty run name, changing it to ", UserWarning)
        pass