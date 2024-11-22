from numpy import float32
import torch
import torchvision
from torch import Tensor, dropout, nn
import torch.nn.functional as F
import math
import torchinfo
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms.functional as fn

#*************************#
# https://www.learnpytorch.io/06_pytorch_transfer_learning/
# https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6
# https://pytorch.org/vision/stable/feature_extraction.html
# https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model
#*************************#

def print_model_summary(model, input_dim:tuple = (32, 3, 224, 224)):
    """
    Function which prints summary of the model.

    Args:
        * model, pytorch model
        * input_dim, tupple, input dimenzions to the model

    Output:
        * returns printable model summary
    """
    return torchinfo.summary(model=model, 
            input_size= input_dim, # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=16,
            row_settings=["var_names"]
            ) 
    
def freeze_model_part(model, freeze_ratio:float):
    """
    Method which freezes part of the model.
    
    Args:
        * model, pytorch model, model which is suppose to be frozen
        * freeze_ratio, float, part of the model which is going to be frozen. 0.8 means that first 
        80% of the layers will be frozen.
    """

    print(f"Freezing ratio: {freeze_ratio}")


    # First bring everything trainable - starting position
    for _param in model.parameters():
            _param.requires_grad = True
            
    # Calculate ratio
    _number_of_layers = len(list(model.named_parameters()))
    _freeze_border = int(freeze_ratio * _number_of_layers)
    
    # Freeze layer
    for _i, _param in enumerate(model.parameters()):
        if _i < _freeze_border:
            _param.requires_grad = False
        
    # Fix bias layer - params + bias must both be frozen
    for _name, _param in model.named_parameters():
        if _param.requires_grad and 'bias' in _name:
            _param.requres_grad = False


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
        
        # RadiologyNet models
        # Check if it is in
        if _src_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_src_name].shape:
                print(f"TRANSFER at layer: {_src_name}/{_src_param.shape}")
                _target_state_dict[_src_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TRANSFER at layer: {_src_name}/{_src_param.shape}")
        else:
            print(f"LAYER NOT FOUND: {_src_name}")
        
        # Expand for ImageNet pretrained models
        _expanded_name = "model."+_src_name
        print(f"TRYING with expaded name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                print(f"TRANSFER at layer: {_expanded_name}/{_src_param.shape}")
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TRANSFER at layer: {_src_name}/{_src_param.shape}")
        
         # Expand for EfficintNet pretrained models
        _expanded_name = _src_name.replace("features", "model")
        print(f"TRYING with swaped name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                print(f"TRANSFER at layer: {_expanded_name}/{_src_param.shape}")
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TRANSFER at layer: {_src_name}/{_src_param.shape}")


        # Go vie versa
        # Short for ImageNet pretrained models
        if len(_src_name.split("model.")) == 1:
            continue
        _shorted_name = _src_name.split("model.")[1]
        print(f"TRYING with shorted name: {_shorted_name}")
        if _shorted_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_shorted_name].shape:
                print(f"TRANSFER at layer: {_shorted_name}/{_src_param.shape}")
                _target_state_dict[_shorted_name].copy_(_src_param)
            else:
                print(f"UNABLE TRANSFER at layer: {_src_name}/{_src_param.shape}")

        else:
            print(f"LAYER NOT FOUND: {_src_name}")
    # Update weights
    target_model.load_state_dict(_target_state_dict)


def freeze_model_base(model, freeze:bool = True, seq = False):
    """
    Script which (un)freezes model's base paramters 

    Args:
        * model, pytorch model
        * freeze, boolean, if True parameters are Frozen, False unfreezes 
        * seq, bool, for some models necessary model then it needs to be set true

    """
    #vit
    if seq == 'vit' and freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False
        model.heads.head.weight.requires_grad = True
        model.heads.head.bias.requires_grad = True
        return 
    
    if seq == 'vit' and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        return 

    # Resnet
    if seq and freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False
        model.model.fc.weight.requires_grad = True
        model.model.fc.bias.requires_grad = True
        return 
    
    if seq and freeze == False:
        for _param in model.parameters():
            _param.requires_grad = True
        return  

    # Efficientnet and VGG
    if seq == False and freeze == True:
        for _param in model.features.parameters():
            _param.requires_grad = False

    if seq == False and freeze == False:
        for _param in model.features.parameters():
            _param.requires_grad = True



class DenseNet121(nn.Module):
    """
    Class for building densnet
    https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

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
        _x = F.sigmoid(_x)
        return _x


class ResNet34(nn.Module):
    """
    Class for building ResNet34 neural network
    """
    def __init__(self, device = "cpu", pretrained = False, number_of_classes = 14):
        """
        Init, start with pretrained weights from imagenet

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
        #_x = self.features(x)
        # For any input image size
        
        _x = self.model(x)
        _x = F.sigmoid(_x)
        return _x
   


#*******************************************************#
# Augumentation models
#*******************************************************#  

class AddChanels:
    """
    Class which simply transform 1 chaneel to 3 channel gray image
    """
    def __init__(self):
        pass
    
    def __call__(self, x):
        # Create 3 channel image
        _image = x.unsqueeze(1)
        _image = _image.repeat(1, 3, 1, 1)
        return _image
    
class TransformGray_GRAZPEDWRI:
    """
    Same as TransformXrayRGB but for gray
    https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full
    https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-data-augmentation
    Class which transform grayscale data by applying:

        * Flipping by the Y axis (reflection): probabiltiy 0.5
        * Random rotation for +- 15 degrees
        * Random brightness transform: 0.8-1.2
        * Random contrast: 0.8-1.3
        * Random saturation: 0.5
        * Random hue: 0.5 
    """
    def __init__(self):
        """
        Define all transofrmations
        """
        self.filp_y = torchvision.transforms.RandomHorizontalFlip(0.5)
        self.random_rotation = torchvision.transforms.RandomRotation(degrees = 15)
        self.image_color_maniupulaiton = torchvision.transforms.ColorJitter(brightness = [0.8, 1.2],
                                           contrast = [0.8 ,1.3],
                                           saturation= 0.5,
                                           hue = 0.5)
        
        self.Add_Chanels = AddChanels()

    def __call__(self, x):
        # Unsqueeze
        _image = torch.unsqueeze(x, 1)
        
        # Flip
        _image = self.filp_y(_image)
        _image = self.random_rotation(_image)
        _image = self.image_color_maniupulaiton(_image)
        
        # Remove channel
        _image = _image.squeeze(1)

        # Apply color invert
        _image = self.Add_Chanels(_image)

        return _image


class TransformToRGB:
    """
    Class which simply transforms grayscale data to RGB input
    """

    def __init__(self):
        pass

    def __call__(self, x):
        # Create 3 channel image
        _image = x.unsqueeze(1)
        _image = _image.repeat(1, 3, 1, 1)
        # Normalize
        _image = fn.normalize(_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Return
        return _image    
    

class TransformRGB_GRAZPEDWRI:
    """
    https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full
    https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-data-augmentation
    Class which transform grayscale data by applying:

        * Flipping by the Y axis (reflection): probabiltiy 0.5
        * Random rotation for +- 15 degrees
        * Random brightness transform: 0.8-1.2
        * Random contrast: 0.8-1.3
        * Random saturation: 0.5
        * Random hue: 0.5 
    """
    def __init__(self):
        """
        Define all transofrmations
        """
        self.filp_y = torchvision.transforms.RandomHorizontalFlip(0.5)
        self.random_rotation = torchvision.transforms.RandomRotation(degrees = 15)
        self.image_color_maniupulaiton = torchvision.transforms.ColorJitter(brightness = [0.8, 1.2],
                                           contrast = [0.8 ,1.3],
                                           saturation= 0.5,
                                           hue = 0.5)
        
        self.RGB_Transform = TransformToRGB()

    def __call__(self, x):
        # Unsqueeze
        _image = torch.unsqueeze(x, 1)
        
        # Flip
        _image = self.filp_y(_image)
        _image = self.random_rotation(_image)
        _image = self.image_color_maniupulaiton(_image)
        
        # Remove channel
        _image = _image.squeeze(1)

        # Apply color invert
        _image = self.RGB_Transform(_image)

        return _image
    

#*******************************************************#
# Utility models
#*******************************************************#  
class lr_wrap_model(nn.Module):
    """
    Model for wrapping preprocesing model with prediction model for finding
    the best learning rate
    """
    def __init__(self, preproc_model, main_model):
        """
        Args:
            * preproc_model, data augumentation model
            * maing_model, main model to test
        """
        super(lr_wrap_model, self).__init__()
        self.preproc = preproc_model
        self.main = main_model
        
    def forward(self, x):
        _x = self.preproc(x)
        _x = self.main(_x)
        return _x
