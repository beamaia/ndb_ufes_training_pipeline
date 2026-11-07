import torch
import torch.nn as nn
from .densenet121 import DenseNet121
from .mobilenet import MobileNetv2
from .efficientnetb4 import EfficientNetB4
from .resnet50 import ResNet50
from .vgg16 import VGG16

# Models built by timm: the name in params.yaml -> the timm name, with the tag of the pretrained weights that is trained from.
TIMM_MODEL_IDS = {
    "coat_lite_small": "coat_lite_small",
    "pit_s_distilled_224": "pit_s_distilled_224",
    "vit_small_patch16_384": "vit_small_patch16_384",
    "convnext_large": "convnext_large.fb_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_base_patch4_window7_224": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
}


# Pathology foundation models, built by timm. The weights are not in this repository: the repositories on Hugging Face are gated, with non-commercial licences, so each
# user needs access with their own HF_TOKEN. "hub" is the repository at a fixed revision. "architecture" and "arguments" build the same network without the hub, which is
# how a checkpoint of ours is loaded again. The arguments are those of the model cards (Virchow2: the config.json and the card; UNI2-h: the card).
FOUNDATION_MODELS = {
    "virchow2": {
        "hub": "hf-hub:paige-ai/Virchow2@3158645804b69e3f3bc4439d4116edddf0840a72",
        "architecture": "vit_huge_patch14_224",
        "arguments": {"img_size": 224, "init_values": 1e-5, "num_classes": 0, "reg_tokens": 4, "mlp_ratio": 5.3375, "global_pool": "", "dynamic_img_size": True},
        "tokens": True,  # the model gives all its tokens; the embedding is the class token and the mean of the patch tokens (the register tokens are left out)
    },
    "uni2_h": {
        "hub": "hf-hub:MahmoodLab/UNI2-h@d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
        "architecture": "vit_giant_patch14_224",
        "arguments": {"img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0,
                      "no_embed_class": True, "reg_tokens": 8, "dynamic_img_size": True},
        "tokens": False,  # the model gives its embedding (the class token)
    },
}


class TimmClassifier(nn.Module):
    def __init__(self, model_id, num_classes, pretrained=True, grad_checkpointing=False):
        super().__init__()
        import timm

        self.model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        if grad_checkpointing:
            self.model.set_grad_checkpointing(True)
        self.features = self.model

    def forward(self, x):
        return self.model(x)

    @property
    def classifier(self):
        return self.model.get_classifier()


class FoundationClassifier(nn.Module):
    """A pathology foundation model, and a linear layer from its embedding of the patch to the classes. Both are trained (end to end), unless the training mode freezes the model."""

    def __init__(self, backbone, num_classes, tokens):
        super().__init__()
        self.backbone, self.tokens = backbone, tokens
        self.features = backbone
        self.classifier = nn.Linear(backbone.num_features * (2 if tokens else 1), num_classes)

    def forward(self, x):
        output = self.backbone(x)
        if self.tokens:  # class token and mean of the patch tokens, without the register tokens
            prefix = self.backbone.num_prefix_tokens
            output = torch.cat([output[:, 0], output[:, prefix:].mean(dim=1)], dim=-1)
        return self.classifier(output)


def build_foundation(name, num_classes, pretrained=True, grad_checkpointing=False):
    """The foundation model ``name`` with its weights from the hub (``pretrained``), or only the network, without contacting the hub."""
    import timm
    from timm.layers import SwiGLUPacked

    spec = FOUNDATION_MODELS[name]
    arguments = {**spec["arguments"], "mlp_layer": SwiGLUPacked, "act_layer": nn.SiLU}  # the models are trained with these; they are not in the configuration of the hub
    backbone = timm.create_model(spec["hub"] if pretrained else spec["architecture"], pretrained=pretrained, **arguments)
    if grad_checkpointing:
        backbone.set_grad_checkpointing(True)
    return FoundationClassifier(backbone, num_classes, spec["tokens"])


class ModelSelector:
    def __init__(self, model_name, weights="default", training_mode="finetune", **kwargs):
        self.model_name = model_name
        self.weights = weights
        self.training_mode = training_mode
        self.model_obj = self._select_model(**kwargs)
        if self.training_mode == "frozen_head":
            self._freeze_backbone()

    def _select_model(self, num_classes, pretrained=True, grad_checkpointing=False, **kwargs):
        """``pretrained=False`` builds the network without its weights and, for the models of timm, without contacting any hub (to load a checkpoint of ours)."""
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
        elif model_name in TIMM_MODEL_IDS:
            model = TimmClassifier(TIMM_MODEL_IDS[model_name], num_classes=num_classes, pretrained=pretrained, grad_checkpointing=grad_checkpointing)
        elif model_name in FOUNDATION_MODELS:
            model = build_foundation(model_name, num_classes, pretrained=pretrained, grad_checkpointing=grad_checkpointing)
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
