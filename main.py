import os
import pyaml
import mlflow

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://localhost:8000/")

    parameters = pyaml.dump("params.yaml")

    experiment_id = mlflow.set_experiment(name="")

    with mlflow.start_run(
        run_id="",
        run_name="",
        experiment_id=experiment_id,
        tags="",
        description="",
        log_system_metrics=True
    ):
        pass