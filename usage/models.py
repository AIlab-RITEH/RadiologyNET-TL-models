import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import logging

logger = logging.getLogger(__name__)

def transfer_weights_to_model(path: str, target_model, device = 'cpu'):
    """
    Function which transfer weights from the model given in path to the target_model
    The weights are transfered only if the name and shape maches
    
    Args:
        * path, str, path to the pytorch saved model
        * target_model, pytroch model, model to which weights will be transfered
        * device, device name, where to map state dict
    """

    # Loading state dicts
    _src_state_dict = torch.load(path, map_location = device)
    _target_state_dict = target_model.state_dict()
    
    # Go trough weights and transfer them
    for _src_name, _src_param in _src_state_dict['model_state'].items():
        
        if _src_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_src_name].shape:
                logger.debug(f"TRANSFER at layer: {_src_name}/{_src_param.shape}")
                _target_state_dict[_src_name].copy_(_src_param)
                continue
            else:
                logger.debug(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")
        else:
            logger.debug(f"LAYER NOT FOUND: {_src_name}")
        
        # Expand for ImageNet pretrained models
        _expanded_name = "model."+_src_name
        logger.debug(f"TRYING with expanded name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                logger.debug(f"TRANSFER at layer: {_expanded_name}/{_src_param.shape}")
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                logger.debug(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")
        
         # Expand for EfficintNet pretrained models
        _expanded_name = _src_name.replace("features", "model")
        logger.debug(f"TRYING with swapped name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                logger.debug(f"TRANSFER at layer: {_expanded_name}/{_src_param.shape}")
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                logger.debug(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")


        # Go vice versa
        # Short for ImageNet pretrained models
        if len(_src_name.split("model.")) == 1:
            continue
        _shorted_name = _src_name.split("model.")[1]
        logger.debug(f"TRYING with shortened name: {_shorted_name}")
        if _shorted_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_shorted_name].shape:
                logger.debug(f"TRANSFER at layer: {_shorted_name}/{_src_param.shape}")
                _target_state_dict[_shorted_name].copy_(_src_param)
            else:
                logger.debug(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")

        else:
            logger.debug(f"LAYER NOT FOUND: {_src_name}")
    # Update weights
    target_model.load_state_dict(_target_state_dict)


class DenseNet121(nn.Module):
    """
    Class for building densnet
    https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """
        # Inherit
        super(DenseNet121, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.DenseNet121_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.densenet121(weights=_weights).to(device)
        
        # Get Features
        self.features = _model.features
        # Edit LastLayer
        _model.classifier = nn.Linear(_model.classifier.in_features, number_of_classes)
        self.classifier = _model.classifier    
        
        # Transfer
        self.model = _model
            
    def forward(self, x):
        _features = self.features(x)
        _out = F.relu(_features, inplace=True)
        _out = F.adaptive_avg_pool2d(_out, (1, 1))
        _out = torch.flatten(_out, 1)
        _x = self.classifier(_out)
        return _x


class InceptionV3(nn.Module):
    """
    Class for building InceptionV3 neural network
    """ 

    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer 
        """

        # Inherent
        super(InceptionV3, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.inception.Inception_V3_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.inception_v3(weights=_weights)
        _model.aux_logits = False
        _model = _model.to(device)

        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)
        
        self.model = _model


    def forward(self, x):
        _x = self.model(x)

        return _x


class EfficientNetB3(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """        Input size: 300x300

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB3, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b3(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1536, number_of_classes) #153600 for ff
        )

    def forward(self, x):
        _x = self.features(x)
        # For any input image size
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x
    

class EfficientNetB4(nn.Module):
    """
    Class for building EfficeientNetB3 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """        Input size: 380x380

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(EfficientNetB4, self).__init__()

        # Load model pretrained on EfficientNet_B3 imagenet
        if pretrained:
            _weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.efficientnet_b4(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(1792, number_of_classes)
        )

    def forward(self, x):
        _x = self.features(x)
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x


class MobileNetV3Small(nn.Module):
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        super(MobileNetV3Small, self).__init__()
        # Load the pretrained MobileNetV3-Small model

        # Load model pretrained on EfficientNet_B2 imagenet
        if pretrained:
            _weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None

        _model = torchvision.models.mobilenet_v3_small(weights=_weights).to(device)
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        # Build head
        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(1024, number_of_classes)
    
    def forward(self, x):
        _x = self.features(x)
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        _x = F.sigmoid(_x)
        return _x


class MobileNetV3Large(nn.Module):
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        super(MobileNetV3Large, self).__init__()
        # Load the pretrained MobileNetV3-Large model
        if pretrained:
            _weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None

        _model = torchvision.models.mobilenet_v3_large(weights=_weights).to(device)
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        # Build head
        self.classifier = _model.classifier
        self.classifier[-1] = nn.Linear(1280, number_of_classes)
    
    def forward(self, x):
        _x = self.features(x)
        _x = self.avgpool(_x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        _x = F.sigmoid(_x)
        return _x

    

class ResNet50(nn.Module):
    """
    Class for building ResNet50 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet50, self).__init__()

        # Load model pretrained on ResNet50 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet50_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet50(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        _x = self.model(x)
        return _x
    


class ResNet34(nn.Module):
    """
    Class for building ResNet34 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet34, self).__init__()

        # Load model pretrained on ResNet34 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet34_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet34(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(512, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        _x = self.model(x)
        return _x
   


class ResNet18(nn.Module):
    """
    Class for building ResNet18 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(ResNet18, self).__init__()

        # Load model pretrained on ResNet18 imagenet
        if pretrained:
            _weights = torchvision.models.ResNet18_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.resnet18(weights=_weights).to(device)
        
        # Edit LastLayer
        _model.fc = nn.Linear(512, number_of_classes)

        # Build model
        self.model = _model

    def forward(self, x):
        _x = self.model(x)
        return _x
   

class VGG16(nn.Module):
    """
    Class for building VGG16 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 10):
        """
        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratch
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(VGG16, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained:
            _weights = torchvision.models.VGG16_Weights.DEFAULT # .DEFAULT = best available weights 
        else: 
            _weights = None
        _model = torchvision.models.vgg16(weights=_weights).to(device)
        
        # Edit feature extractors
        self.features = _model.features
        self.avgpool = _model.avgpool
        
        # Build head
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, number_of_classes)
        )

    def forward(self, x):
        _x = self.features(x)
        _x = torch.flatten(_x, 1)
        _x = self.classifier(_x)
        return _x

