"""A checkpoint that describes itself: the weights and everything needed to rebuild the model and use it on new images."""

import torch

from src.models.model_selector import ModelSelector


def save_checkpoint(path, model, description: dict) -> None:
    """``description`` holds only plain values (str, int, float, list, dict), so the file loads with ``weights_only=True``."""
    torch.save({"model_state_dict": model.state_dict(), **description}, path)


def load_checkpoint(path, device, node_type):
    """The model rebuilt from the checkpoint, ready on ``device``, and the description saved with it."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = bundle.pop("model_state_dict")
    model = ModelSelector(bundle["model_name"], num_classes=len(bundle["classes"]), training_mode=bundle["training_mode"]).model
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).to(node_type), bundle

