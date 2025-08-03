from torchvision import models
from .densenet121 import DenseNet121
from .mobilenet import MobileNetv2
from .efficientnetb4 import EfficientNetB4
from .resnet50 import ResNet50
from .vgg16 import VGG16

class ModelSelector:
    def __init__(self, model_name, weights="default", **kwargs):
        self.model_name = model_name
        self.weights = weights
        self.model_obj = self._select_model(**kwargs)

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
        else:
            raise ValueError(f"String value {model_name} invalid for choosing model type.")
        
        return model
    
    @property
    def model(self):
        return self.model_obj
    
    @property
    def model_features(self):
        return self.model_obj.features
    
    @property
    def model_classifier_layer(self):
        return self.model_obj.classifier