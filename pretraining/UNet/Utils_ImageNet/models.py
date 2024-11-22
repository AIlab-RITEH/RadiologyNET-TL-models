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
            col_width=20,
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


def transfer_weights_to_model(path: str, target_model):
    """
    Function which transfer weights from the model given in path to the target_model
    The weights are transfered only if the name and shape maches
    
    Args:
        * path, str, path to the pytorch saved model
        * target_model, pytroch model, model to which weights will be transfered
    """

    # Loading state dicts
    _src_state_dict = torch.load(path)
    _target_state_dict = target_model.state_dict()
    
    # Go trough weights and transfer them
    for _src_name, _src_param in _src_state_dict['model_state'].items():
        # Check if it is in
        if _src_name in _target_state_dict:
            # Name exist, but is the shape correct
            if _src_param.shape == _target_state_dict[_src_name].shape:
                _target_state_dict[_src_name].copy_(_src_param)
                print(f"TRANSFER SUCCESSFULL at layer: {_src_name}/{_src_param.shape}")
            else:
                print(f"UNABLE TRANSFER at layer: {_src_name}/{_src_param.shape}")
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
    # UNet
    if freeze:
        freeze = False
    else:
        freeze = True
    	
    for _param in model.parameters():
        _param.requires_grad = freeze
        

class InceptionV3(nn.Module):
    """
    Class for building InceptionV3 neural network
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
        super(InceptionV3, self).__init__()

        # Load model pretrained on VGG16 imagenet
        if pretrained: 
            _weights = torchvision.models.inception.Inception_V3_Weights.DEFAULT  # .DEFAULT = best available weights 
        else: 
            _weights = None
        
        _model = torchvision.models.inception_v3(weights=_weights).to(device)

        # Edit LastLayer
        _model.fc = nn.Linear(2048, number_of_classes)
        
        self.model = _model


    def forward(self, x):
        _x = self.model(x)

        return _x
    

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.final = nn.Sigmoid() 

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        
        x = self.last(x)
        x = self.final(x)

        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



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
    
class TransformXrayGray:
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
    

class TransformXrayRGB:
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
