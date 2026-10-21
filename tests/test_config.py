from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import deep_merge, load_params
from src.validate_input.params import Parameters

REPO = Path(__file__).resolve().parents[1]


def write(path, settings):
    path.write_text(yaml.safe_dump(settings))
    return path


def test_the_override_wins_down_to_a_single_setting():
    assert deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}, "c": 4}) == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_the_protocol_file_loads_and_paths_start_at_the_organizer_repository():
    params, files = load_params(REPO / "params.yaml", REPO)
    organizer = (REPO / "../ndb_ufes_data_organizer").resolve()
    assert params.dataset.folds_csv == str(organizer / "results/phase1/folds/fold_assignments.csv")
    assert params.dataset.patch_dir == str(organizer / "data/ndb_ufes/patch_level/images")
    assert params.run.purpose == "final" and params.hyperparameters.other.epochs == 150
    assert files == [(REPO / "params.yaml").resolve()]


def test_the_trial_file_extends_the_protocol_and_changes_only_what_it_names():
    final, _ = load_params(REPO / "params.yaml", REPO)
    trial, files = load_params(REPO / "configs/trial.yaml", REPO)
    assert trial.run.purpose == "trial" and trial.hyperparameters.other.epochs == 2
    assert trial.hyperparameters.optimizer == final.hyperparameters.optimizer and trial.seed == final.seed
    assert files == [(REPO / "params.yaml").resolve(), (REPO / "configs/trial.yaml").resolve()]


def test_the_merged_settings_can_be_saved_and_loaded_again(tmp_path):
    params, _ = load_params(REPO / "configs/trial.yaml", REPO)
    saved = write(tmp_path / "merged.yaml", params.model_dump(mode="json"))
    assert Parameters.model_validate(yaml.safe_load(saved.read_text())) == params


def test_an_absolute_organizer_and_absolute_paths_are_kept(tmp_path):
    machine = write(tmp_path / "machine.yaml", {"extends": str(REPO / "params.yaml"), "dataset": {"organizer": "/data/organizer", "folds_csv": "/elsewhere/folds.csv"}, "device": "cpu"})
    params, _ = load_params(machine, REPO)
    assert params.dataset.folds_csv == "/elsewhere/folds.csv"
    assert params.dataset.patch_dir == "/data/organizer/data/ndb_ufes/patch_level/images"
    assert params.device == "cpu"


def test_an_unknown_setting_in_a_file_is_an_error(tmp_path):
    typo = write(tmp_path / "typo.yaml", {"extends": str(REPO / "params.yaml"), "run": {"purpse": "trial"}})
    with pytest.raises(ValidationError):
        load_params(typo, REPO)


def test_files_that_extend_each_other_in_a_circle_are_refused(tmp_path):
    write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="circle"):
        load_params(tmp_path / "a.yaml", REPO)


def test_the_final_file_runs_every_model_with_the_protocol_and_a_cost_limit():
    protocol, _ = load_params(REPO / "params.yaml", REPO)
    final, _ = load_params(REPO / "configs/final.yaml", REPO)
    assert len(final.experiment.models) == len(set(final.experiment.models)) == 12
    assert final.hyperparameters == protocol.hyperparameters and final.run.purpose == "exp2"
    assert final.run.max_cost_usd == 20 and final.run.hourly_cost == 0.59


def test_a_cost_limit_needs_a_price(tmp_path):
    with pytest.raises(ValidationError, match="hourly_cost"):
        load_params(write(tmp_path / "limit_without_price.yaml", {"extends": str(REPO / "params.yaml"), "run": {"max_cost_usd": 5}}), REPO)


def test_deleting_checkpoints_needs_the_upload(tmp_path):
    with pytest.raises(ValidationError, match="checkpoints_in_mlflow"):
        load_params(write(tmp_path / "delete_without_upload.yaml", {"extends": str(REPO / "params.yaml"), "run": {"delete_checkpoints_after_upload": True}}), REPO)


def test_experiment_1_changes_only_the_folds_file_and_the_experiment_name():
    final, _ = load_params(REPO / "configs/final.yaml", REPO)
    first, _ = load_params(REPO / "configs/experiment_1.yaml", REPO)
    assert first.dataset.folds_csv.endswith("fold_assignments_published.csv") and first.run.purpose == "exp1"
    assert first.dataset.test_fold == final.dataset.test_fold == 5
    assert first.experiment.models == final.experiment.models and first.hyperparameters == final.hyperparameters


def test_the_resumed_experiment_1_runs_the_models_that_were_not_finished_with_a_lower_limit():
    full, _ = load_params(REPO / "configs/experiment_1.yaml", REPO)
    resumed, _ = load_params(REPO / "configs/experiment_1_resume.yaml", REPO)
    assert resumed.experiment.models == full.experiment.models[2:]  # mobilenetv2 and efficientnet_b0 were finished
    assert resumed.run.purpose == "exp1" and resumed.dataset.folds_csv == full.dataset.folds_csv and resumed.run.max_cost_usd == 11
