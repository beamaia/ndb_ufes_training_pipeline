import os
import yaml
import mlflow

from src.dataset import NDBUfesOrganizer, NDBUfesDataset
from src.pipeline import Pipeline
from src.validate_input.params import Parameters
from src import logger, log_file

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://localhost:8000/")

    if not os.path.exists("params.yaml"):
        raise FileNotFoundError("File params.yaml does not exist. Please create params.yaml file in order to run the project.")

    with open("params.yaml", "r") as f:
        params_dict= yaml.load(f, yaml.Loader)
    params_model = Parameters.model_validate(params_dict)

    logger.info(params_model)

    data_organizer = NDBUfesOrganizer(task=params_model.project.value,  **params_model.dataset.model_dump())

    experiment_id = mlflow.set_experiment(experiment_name=params_model.project.value)

    with mlflow.start_run(
        run_name=params_model.run_name,
        experiment_id=experiment_id.experiment_id,
        tags="",
        description="",
        log_system_metrics=True
    ):
        pipeline = Pipeline(params_model, data_organizer)
        pipeline.log_params()
        
        if params_model.stages.train == True:
            pipeline.train()

        mlflow.log_artifact(log_file)
    