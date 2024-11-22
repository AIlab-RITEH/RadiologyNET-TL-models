from numpy import float32
import torchvision
import torch
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
            col_width=20,
            row_settings=["var_names"]
            ) 

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


def freeze_model_base(model, freeze:bool = True):
    """
    Script which (un)freezes model's base paramters 

    Args:
        * model, pytorch model
        * freeze, boolean, if True parameters are Frozen, False unfreezes 
    
    """

    # Freeze 
    if freeze == True:
        for _param in model.parameters():
            _param.requires_grad = False

        # This is for gender age
        #model.fc_branch.weight.requires_grad = True
        #model.fc_branch.bias.requires_grad = True

        for _param in model.sequential_head.parameters():
            _param.requires_grad = True
    else:
        for _param in model.parameters():
            _param.requires_grad = True

        


def get_model_output_shape(model:nn.Module, input_channels:int = 3, input_size:int = 500)->int:
    """
    Get number of output neurons for a given model

    Args:
        * model, nn.module, model for which we are interested to find its output
        * input shape, int, input image with shape (1, input_channels, input_size, input_size)
    """
    _sample_input = torch.randn(1, input_channels, input_size, input_size)  # Define a sample input tensor
    
    model.eval()
    # Run the sample input through the model
    with torch.no_grad():
        _output = model(_sample_input)
    
    model.train()
    # Output
    _output = torch.flatten(_output)

    # Return value
    return _output.shape[0]


class RSNA_model(nn.Module):
    """
    Class for building RSNA test model

    DISCLAIMER: THEY HAVE REPORTED 100,000 OUTPUTS AT CONCATENATION... GUESS WHAT, I'VE GOT 400000, SO I HAVE SCALED EVERYTING.
    """
    def __init__(self, device = "cpu", backbone = 'inceptionV3', pretrained = False, input_dim = 500):
        """
        Init, start with pretrained weights from imagenet

        Args:
            * device, str, "cpu" or "cuda" or device id where model will be located
            * pretrained, boolean, True to load best possible weights, False for training from scratcth
            * number_of_classes, int, number of neurons in last layer
            
        """
        # Inherent
        super(RSNA_model, self).__init__()

        # Load model pretrained on Inception_v3 imagenet
        
        if backbone == 'inceptionV3':
            if pretrained: 
                _weights = torchvision.models.inception.Inception_V3_Weights.DEFAULT  # .DEFAULT = best available weights 
            else: 
                _weights = None
            
            _model = torchvision.models.inception_v3(weights=_weights).to(device)
            _model.aux_logits = False
        
            # Edit LastLayer Necessary for transfer learning
            _model.avgpool = nn.Identity()
            _model.dropout = nn.Identity()
            _model.fc = nn.Identity()
            self.model = _model

        if backbone == 'resNet50':
            # Load model pretrained on ResNet50 imagenet
            ##  WANRNING: not tested yet !!
            if pretrained:
                _weights = torchvision.models.ResNet50_Weights.DEFAULT # .DEFAULT = best available weights 
            else: 
                _weights = None
            _model = torchvision.models.resnet50(weights=_weights).to(device)

             # Edit LastLayer Necessary for transfer learning
            _model.avgpool = nn.Identity()
            _model.fc = nn.Identity()
            self.model = _model

        if backbone == 'eff3':
            # Load model pretrained on EfficientNet_B3 imagenet. Input size: 300x300
            if pretrained:
                _weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT # .DEFAULT = best available weights 
            else: 
                _weights = None
            _model = torchvision.models.efficientnet_b3(weights=_weights).to(device)
            
            # LastLayer Necessary for transfer learning
            self.model = _model.features
                        

        # Flatten layer
        self.Flatten = nn.Flatten()

        # Obtain otuput size
        _output_number_of_neurons = get_model_output_shape(self.model, input_size = input_dim)

        # Add branch for age
        _scale = 4 # to compensate difference in number of layers
        self.fc_branch = nn.Linear(1, 32 * _scale)
        # Create head
        _input_number_neurons = _output_number_of_neurons + _scale * 32
        self.sequential_head = nn.Sequential(
            nn.Linear(_input_number_neurons, 1000),  # Input size: input_size, Output size: hidden_size1
            nn.ReLU(),
            nn.Linear(1000, 1000),  # Input size: hidden_size1, Output size: hidden_size2
            nn.ReLU(),
            nn.Linear(1000, 1)  # Input size: hidden_size2, Output size: output_size
        )
  
  

    def forward(self, x, y):
        # Image branch
        _x = self.model(x)

        #_x = _x.logits
        _x = self.Flatten(_x)
        
        # Age branch
        _y = self.fc_branch(y)
        # Concatenation
        _concatentated = torch.cat((_x, _y), dim = 1)

        # Head
        _out = self.sequential_head(_concatentated)
        return _out
  
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
    

class TransformRGB_RSNA:
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
        self.random_rotation = torchvision.transforms.RandomRotation(degrees = 20)
        self.translate = torchvision.transforms.RandomAffine(0,              # No shear
                             (0.2, 0.2),    # Horizontal and vertical translation up to 20%
                             scale=(0.8, 1.2))
        
        self.RGB_Transform = TransformToRGB()

    def __call__(self, x):
        # Unsqueeze
        _image = torch.unsqueeze(x, 1)
        
        # Flip
        _image = self.filp_y(_image)
        _image = self.random_rotation(_image)
        _image = self.translate(_image)
        
        # Remove channel
        _image = _image.squeeze(1)

        # Apply color invert
        _image = self.RGB_Transform(_image)

        return _image
        
class TransformGray_RSNA:
    """
    https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full
    https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-data-augmentation
    https://pubs.rsna.org/doi/suppl/10.1148/radiol.2018180736/suppl_file/ry180736suppa1.pdf
    Class which transform grayscale data by applying:

        * Flipping by the Y axis (reflection): probabiltiy 0.5
        * Random rotation for +- 20 degrees
        * Horizontal vertical translation to 20%
        * Zoom up to 20%
        * horizontal flip 
    """
    def __init__(self):
        """
        Define all transofrmations
        """
        self.filp_y = torchvision.transforms.RandomHorizontalFlip(0.5)
        self.random_rotation = torchvision.transforms.RandomRotation(degrees = 20)
        self.translate = torchvision.transforms.RandomAffine(0,              # No shear
                             (0.2, 0.2),    # Horizontal and vertical translation up to 20%
                             scale=(0.8, 1.2))
        
        self.Add_Chanels = AddChanels()

    def __call__(self, x):
        # Unsqueeze
        _image = torch.unsqueeze(x, 1)
        
        # Flip
        _image = self.filp_y(_image)
        _image = self.random_rotation(_image)
        _image = self.translate(_image)
        
        # Remove channel
        _image = _image.squeeze(1)

        # Apply color invert
        _image = self.Add_Chanels(_image)

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
