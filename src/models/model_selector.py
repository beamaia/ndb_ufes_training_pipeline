import torch
import torch.nn as nn
from torchvision import models
from .densenet121 import DenseNet121
from .mobilenet import MobileNetv2
from .efficientnetb4 import EfficientNetB4
from .resnet50 import ResNet50
from .vgg16 import VGG16

HF_MODEL_IDS = {
    "uni": "MahmoodLab/UNI",
    "virchow": "paige-ai/Virchow",
    "ctranspath": "kaczmarj/CTransPath",
    "mocov3_vit_small": "1aurent/vit_small_patch16_224.transpath_mocov3",
    "vit_base_patch16_224": "google/vit-base-patch16-224",
    "vit_large_patch16_224": "google/vit-large-patch16-224",
    "vit_base_patch32_224": "google/vit-base-patch32-224",
    "deit_base_patch16_224": "facebook/deit-base-patch16-224",
    "swin_base_patch4_window7_224": "microsoft/swin-base-patch4-window7-224",
}

TORCHVISION_MODEL_IDS = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
}

TIMM_MODEL_IDS = {
    "coat_lite_small",
    "pit_s_distilled_224",
    "vit_small_patch16_384",
}


class HuggingFaceClassifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install `transformers` to train Hugging Face registry models.") from exc

        try:
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load Hugging Face model '{model_id}'. "
                "Check network/cache availability, credentials, and accepted model terms."
            ) from exc

        hidden_size = self._infer_hidden_size()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.features = self.backbone
        self.prefer_pooler = "swin" in model_id.lower()

    def _infer_hidden_size(self):
        config = getattr(self.backbone, "config", None)
        for attr in ("hidden_size", "projection_dim", "embed_dim", "num_features"):
            value = getattr(config, attr, None)
            if value:
                return int(value)
        if hasattr(config, "hidden_sizes") and config.hidden_sizes:
            return int(config.hidden_sizes[-1])
        raise ValueError("Could not infer Hugging Face backbone hidden size.")

    def _pool_outputs(self, outputs):
        if self.prefer_pooler and hasattr(outputs, "pooler_output"):
            if outputs.pooler_output is not None:
                return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            if hidden.ndim == 3:
                return hidden[:, 0]
            if hidden.ndim == 4:
                return hidden.mean(dim=(2, 3))
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if isinstance(outputs, (tuple, list)):
            first = outputs[0]
            if first.ndim == 3:
                return first[:, 0]
            if first.ndim == 4:
                return first.mean(dim=(2, 3))
            return first
        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                return outputs[:, 0]
            if outputs.ndim == 4:
                return outputs.mean(dim=(2, 3))
            return outputs
        raise ValueError("Unsupported Hugging Face output format.")

    def forward(self, x):
        try:
            outputs = self.backbone(pixel_values=x)
        except TypeError:
            outputs = self.backbone(x)
        features = self._pool_outputs(outputs)
        return self.classifier(features)


class TorchvisionClassifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        if not hasattr(models, model_id):
            raise ValueError(f"torchvision model '{model_id}' is not available.")
        self.model = getattr(models, model_id)(weights="DEFAULT")
        self.features = getattr(self.model, "features", self.model)
        self._replace_classifier(num_classes)

    def _replace_classifier(self, num_classes):
        if hasattr(self.model, "classifier"):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Sequential):
                in_features = classifier[-1].in_features
                classifier[-1] = nn.Linear(in_features, num_classes)
                self.classifier = classifier
            else:
                in_features = classifier.in_features
                self.model.classifier = nn.Linear(in_features, num_classes)
                self.classifier = self.model.classifier
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            self.classifier = self.model.fc
        else:
            raise ValueError(f"Do not know how to replace classifier for {self.model.__class__.__name__}")

    def forward(self, x):
        return self.model(x)


class TimmClassifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install `timm` to train the article transformer models.") from exc

        self.model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=num_classes,
        )
        self.features = self.model

    def forward(self, x):
        return self.model(x)

    @property
    def classifier(self):
        return self.model.get_classifier()


class ModelSelector:
    def __init__(self, model_name, weights="default", training_mode="finetune", **kwargs):
        self.model_name = model_name
        self.weights = weights
        self.training_mode = training_mode
        self.model_obj = self._select_model(**kwargs)
        if self.training_mode == "frozen_head":
            self._freeze_backbone()

    def _select_model(self, num_classes, **kwargs):
        model_name = self.model_name.lower()
        if model_name == 'densenet121':
            model =  DenseNet121(num_classes=num_classes)
        elif model_name == "mobilenetv2":
            model = MobileNetv2(num_classes=num_classes)
        elif model_name == "efficientnetb4":
            model = EfficientNetB4(num_classes=num_classes)
        elif model_name == "vgg16":
            model = VGG16(num_classes=num_classes)
        elif model_name== "resnet50":
            model = ResNet50(num_classes=num_classes)
        elif model_name in TORCHVISION_MODEL_IDS:
            model = TorchvisionClassifier(TORCHVISION_MODEL_IDS[model_name], num_classes=num_classes)
        elif model_name in TIMM_MODEL_IDS:
            model = TimmClassifier(model_name, num_classes=num_classes)
        elif model_name in HF_MODEL_IDS:
            model = HuggingFaceClassifier(HF_MODEL_IDS[model_name], num_classes=num_classes)
        else:
            raise ValueError(f"String value {model_name} invalid for choosing model type.")
        
        return model

    def _freeze_backbone(self):
        for parameter in self.model_obj.parameters():
            parameter.requires_grad = False
        classifier = getattr(self.model_obj, "classifier", None)
        if classifier is None:
            raise ValueError(f"Model {self.model_name} does not expose a classifier for frozen_head mode.")
        classifier_modules = classifier if isinstance(classifier, (tuple, list)) else [classifier]
        for module in classifier_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    
    @property
    def model(self):
        return self.model_obj
    
    @property
    def model_features(self):
        return self.model_obj.features
    
    @property
    def model_classifier_layer(self):
        return self.model_obj.classifier
