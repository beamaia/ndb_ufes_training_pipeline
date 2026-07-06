import os
import yaml
import mlflow
from dotenv import load_dotenv 
load_dotenv()
load_dotenv(os.getenv("MLFLOW_ENV_FILE", "/Volumes/ssd/thesis_organization/ndb_ufes_mlflow/.env"), override=False)

from src.dataset import LeakageSafeNDBUfesOrganizer
from src.pipeline import Pipeline
from src.validate_input.params import Parameters
from src import logger, log_file

if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:8000/")
    os.environ.setdefault("DVC_SITE_CACHE_DIR", ".dvc/tmp/site_cache")
    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))

    if not os.path.exists("params.yaml"):
        raise FileNotFoundError("File params.yaml does not exist. Please create params.yaml file in order to run the project.")

    with open("params.yaml", "r") as f:
        params_dict= yaml.load(f, yaml.Loader)
    params_model = Parameters.model_validate(params_dict)

    experiment_id = mlflow.set_experiment(experiment_name=params_model.project.value)
    model_names = params_model.experiment.models or [params_model.hyperparameters.model.name]

    for model_name in model_names:
        run_params = params_model.model_copy(deep=True)
        run_params.hyperparameters.model.name = model_name
        if len(model_names) > 1:
            run_params.run_name = f"{model_name.value}_{run_params.run_name}"

        logger.info(run_params)
        data_organizer = LeakageSafeNDBUfesOrganizer(run_params)

        with mlflow.start_run(
            run_name=run_params.run_name,
            experiment_id=experiment_id.experiment_id,
            tags={"model": model_name.value, "task": "multiclass"},
            description="Leakage-safe multiclass patch training with fold 5 held out.",
            log_system_metrics=True
        ):
            pipeline = Pipeline(run_params, data_organizer)
            pipeline.log_params()
            
            if run_params.stages.train:
                pipeline.train()

            if run_params.stages.test:    
                pipeline.test()

            mlflow.log_artifact(log_file)
