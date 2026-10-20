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


class TimmClassifier(nn.Module):
    def __init__(self, model_id, num_classes, pretrained=True):
        super().__init__()
        import timm

        self.model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
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
        elif model_name in TIMM_MODEL_IDS:
            model = TimmClassifier(TIMM_MODEL_IDS[model_name], num_classes=num_classes)
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
