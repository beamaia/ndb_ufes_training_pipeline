from pathlib import Path

import torch
import yaml

from src.validate_input.params import Parameters

PATH_KEYS = ("folds_csv", "patch_dir", "patch_table")  # in the dataset section; a relative path starts at the data organizer repository


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    for device, available in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())):
        if available:
            return device
    return "cpu"


def read_settings(file: Path, seen: tuple[Path, ...] = ()) -> tuple[dict, list[Path]]:
    """The settings of a YAML file, and the files they come from. A file can say ``extends: other.yaml`` (a path relative to it): it then starts from the
    settings of that file and changes only what it names."""
    file = Path(file).resolve()
    if file in seen:
        raise ValueError(f"The settings files extend each other in a circle: {' -> '.join(str(f) for f in (*seen, file))}")
    settings = yaml.safe_load(file.read_text()) or {}
    parent = settings.pop("extends", None)
    if parent is None:
        return settings, [file]
    base, files = read_settings(file.parent / parent, (*seen, file))
    return deep_merge(base, settings), [*files, file]


def load_params(file: Path, repo: Path, resume: bool = False) -> tuple[Parameters, list[Path]]:
    """The checked settings of a run, and the YAML files they come from (the first is the base). ``dataset.organizer`` is the folder of the data organizer repository (relative to ``repo``)."""
    settings, files = read_settings(file)
    dataset = settings.setdefault("dataset", {})
    organizer = Path(dataset.pop("organizer"))
    organizer = organizer if organizer.is_absolute() else (repo / organizer).resolve()
    for key in PATH_KEYS:
        if dataset.get(key) and not Path(dataset[key]).is_absolute():
            dataset[key] = str(organizer / dataset[key])
    settings["device"] = resolve_device(settings.get("device", "auto"))
    settings["resume"] = resume
    return Parameters.model_validate(settings), files
