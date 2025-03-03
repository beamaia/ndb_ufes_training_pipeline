from torchvision import models
from .densenet121 import DenseNet121

class ModelSelector:
    def __init__(self, model_name, weights="default", **kwargs):
        self.model_name = model_name
        self.weights = weights
        self.model_obj = self._select_model(**kwargs)

    def _select_model(self, num_classes, **kwargs):
        model_name = self.model_name.lower()
        if model_name == 'densenet121':
            model =  DenseNet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1, num_classes=num_classes)
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