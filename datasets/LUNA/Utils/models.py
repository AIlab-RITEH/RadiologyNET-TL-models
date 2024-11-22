import torch
import torch.nn as nn
import torchvision
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
import random

#*******************************************************#
# Methods which serves as utilities functions
#*******************************************************#

def freeze_model_base(model, freeze:bool = True):
    """
    Script which (un)freezes model's base paramters 

    Args:
        * model, pytorch model
        * freeze, boolean, if True parameters are Frozen, False unfreezes 
        * seq, bool, for some models necessary model then it needs to be set true

    """
    if freeze == True:
        for _param in model.down_blocks.parameters():
            _param.requires_grad = False
    else:
        for _param in model.down_blocks.parameters():
            _param.requires_grad = True
            
            
            
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

def get_model_output_shape(model:nn.Module, input_channels:int = 3, input_size:int = 500)->int:
    """
    Get number of output neurons for a given model

    Args:
        * model, nn.module, model for which we are interested to find its output
        * input shape, int, input image with shape (1, input_channels, input_size, input_size)
    """
    _sample_input = torch.randn(1, input_channels, input_size, input_size)  # Define a sample input tensor
    # Run the sample input through the model
    model.eval()
    with torch.no_grad():
        _output = model(_sample_input)
    model.train()

    # Output
    #_output = torch.flatten(_output)

    # Return value
    return _output
 
def print_model_summary(model, input_dim:tuple = (32, 3, 224, 224), device = "cpu"):
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
            row_settings=["var_names"],
            device=device
    ) 

def split_into_sequentials(model):

    """
    It is much easier to work with the model if it is split into sequentials. Some models
    requires to be splited into sequentials. One example is VGG (Visual Geometric Group) family of
    models. What the function does is simple: Split model in sequnece of sequential block where
    the begginign of each block is MaxPoolingLayer. 

    Args:
        * model, nn.Model, model which wants to be make into sequence

    Output:
        * new_model, nn.Model, sequential version of the model.
    """
    _blocks = []
    _sequential_list = []
    for _layer in list(model.features.children()):
        #print(layer)
        if isinstance(_layer, nn.MaxPool2d):
            _sequential_list.append(nn.Sequential(*_blocks))
            _blocks = []
            _blocks.append(_layer)
        else:
            _blocks.append(_layer)
            
    _new_model = nn.Sequential(*_sequential_list)
    return _new_model

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

#*******************************************************#
# U-Net building blocks
#*******************************************************# 
class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some

    Args:
        * in_channels, int, input size to convoluions in the ConvBlock
        * out_channels, int, output size of the convolution block ConvBlock (last one)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample/1x1 Conv -> ConvBlock -> ConvBlock
    
    Args:
        * in_channels, int, input size to convoluions in the ConvBlock
        * out_channels, int, output size of the convolution block ConvBlock (last one)
        * up_conv_in_channels, int, input size of the upsampling convlution layer
        * up_conv_out_channels, int, output size of the upsampling convolution layer
        * upsample, bool, if the upsample is false then 1x1 conv is done instead of upsampleing. Sometimes
            this is neccesary to fix the number of channels in layers for concatenation.
    """
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose", upsample = True):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

        self.upsample_flag = upsample
        self.conv1x1 = nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        """
        if self.upsample_flag:
            x = self.upsample(up_x)
        else:
            x = self.conv1x1(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    
class UNet(nn.Module):
    """
    Class which implemets U-net for the seleceted backbones.

    Args:
        * device, string, where the model will be loaded. Default is CPU
        * weights_path, string/path, path to model weights in case we want to transfer weights from 
            anoteher pretred model
        * pretrained, bool, flag which marks if the model is pretrained on the ImageNet (no need to provide
            weights path to ImageNet pretrained model since it is a built in mechanism)
        * n_classes, int, number of output channels. Defualt s 1, but if someone have multiple objects,
          it is possible to output multiple channels
        * backbone, string, representing the backbone neural network. Currently implemented: res50, vgg16, effB4.
          it can work wit little to non work with other ResNet/eff/vgg family. 

    Output:
        * U-net model, nn.model.
    """
   
    def __init__(self, device = "cpu", weights_path = None, pretrained = False, n_classes=1, backbone = 'res50'):
        super().__init__()

        # If pretrained - lopad image net weights
        if pretrained == True:
            if backbone == 'res50':
                _weights = torchvision.models.ResNet50_Weights.DEFAULT
            if backbone == 'vgg16':
                _weights = torchvision.models.VGG16_Weights.DEFAULT
            if backbone == 'eff4':
                _weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        else:
            # Models weights are none - random
            _weights = None

        # Create backbone model
        if backbone == 'res50':
            _model = torchvision.models.resnet50(weights = _weights).to(device)
        if backbone == 'vgg16':
            _model = torchvision.models.vgg16(weights = _weights).to(device)
        if backbone == 'eff4':
            _model = torchvision.models.efficientnet_b4(weights = _weights).to(device)

        # Init random weights of the model if the model does not have pretrained weigths
        if pretrained == False:
            self._init_weights()

        # Transfer weights if necessary
        if weights_path != None:
            transfer_weights_to_model(weights_path, _model)
                
        # Storage
        self.down_blocks = []
        _down_channels = []
        self.up_blocks = []

        # Placeholders for preprocessing
        self.input_block = None
        self.input_pool = None
        
        # Buliding resnet encoder
        if backbone in ['res50']:
            # Obtain blocks / modules from which the skip connections will be made
            self.input_block = nn.Sequential(*list(_model.children()))[:3]
            self.input_pool = list(_model.children())[3]
            self.input_block.to(device)
            self.input_pool.to(device)
            
            for bottleneck in list(_model.children()):
                if isinstance(bottleneck, nn.Sequential):
                    self.down_blocks.append(bottleneck)
            self.down_blocks = nn.ModuleList(self.down_blocks)
        
            # Obtain channels in blocks where the skip connections will be made
            # Dummy variable
            _x = torch.rand(1, 3 ,224,224).to(device)
            _down_channels.append(_x.shape)
            _x = self.input_block(_x)
            _down_channels.append(_x.shape)
            _x = self.input_pool(_x)
            _model.eval()

            # Go trough layers and obtain shapes
            for _i, _layer in enumerate(_model.children()):
                # Skip first four layers
                if _i < 4:
                    continue
                # Obtain value
                _x = _layer(_x)
                if isinstance(_layer, nn.Sequential):
                    _down_channels.append(_x.shape)
                
                # Flatten is not in model, but it is required for linear - so we add it
                if isinstance(_layer, nn.AdaptiveAvgPool2d):
                    _x = torch.flatten(_x)
            _model.train()
        
        # Bulding efficient net encoder
        if backbone == 'eff4':
            # Add identity
            self.input_block = nn.Identity()
            
            # Obtain blocks / modules from which the skip connections will be made
            for bottleneck in list(_model.features.children()):
                    self.down_blocks.append(bottleneck)
            self.down_blocks = nn.ModuleList(self.down_blocks)
            
            # Obtain channels in blocks where the skip connections will be made
            # Dummy variable
            _x = torch.rand(1, 3,224,224)
            _down_channels.append(_x.shape)
            _model.eval()
            # Go trough layers and obtain shapes
            for _i, _layer in enumerate(_model.features.children()):
                # Obtain value
                _x = _layer(_x)
                _down_channels.append(_x.shape)
                
                # Flatten is not in model, but it is required for linear - so we add it
                if isinstance(_layer, nn.AdaptiveAvgPool2d):
                    _x = torch.flatten(_x)
            _model.train()

        # Bulding vgg encoder
        if backbone == 'vgg16':
            # Add identity
            self.input_block = nn.Identity()

            # VGG network must be splitted into sequential blocks
            _model = split_into_sequentials(_model)
            
            # Obtain blocks / modules from which the skip connections will be made
            for bottleneck in list(_model.children()):
                #if isinstance(bottleneck, nn.Sequential):
                self.down_blocks.append(bottleneck)
            self.down_blocks = nn.ModuleList(self.down_blocks)
            
            # Obtain channels in blocks where the skip connections will be made
            # Dummy variable
            _x = torch.rand(1, 3,224,224).to(device)
            _down_channels.append(_x.shape)
            _model.eval()
            # Go trough layers and obtain shapes
            for _i, _layer in enumerate(_model.children()):
                # Obtain value
                _x = _layer(_x)
                _down_channels.append(_x.shape)
                
                # Flatten is not in model, but it is required for linear - so we add it
                if isinstance(_layer, nn.AdaptiveAvgPool2d):
                    _x = torch.flatten(_x)
            _model.train()
        
        # Set number of channels for bridge
        _bridge_shape_channels = _down_channels[-1][1]

        # Create bridge
        self.bridge = Bridge(_bridge_shape_channels, _bridge_shape_channels)
            
        # Create up_blocks
        _reverse_down_channels = _down_channels.copy()
        _reverse_down_channels.reverse()    

        # Go trugh channels and build blocks
        for _block_cnt in range(0,len(_reverse_down_channels)):
            # Zero layer is examption - it is the layer near bridge
            if _block_cnt == 0:
                _up_block = UpBlock(in_channels = _reverse_down_channels[_block_cnt][1] * 2, 
                      out_channels = _reverse_down_channels[_block_cnt][1], 
                      up_conv_in_channels= _reverse_down_channels[_block_cnt][1], 
                      up_conv_out_channels= _reverse_down_channels[_block_cnt][1],
                               upsample = False)                
            else:
                # It is neccessary to check if the upsampling is present or not(channels and sizes must macth in
                # order to concatenate them)
                if _reverse_down_channels[_block_cnt][2] == _reverse_down_channels[_block_cnt-1][2]:
                    _upsample = False
                else:
                    _upsample = True
                _up_block = UpBlock(in_channels = _reverse_down_channels[_block_cnt][1] * 2, 
                          out_channels = _reverse_down_channels[_block_cnt][1], 
                          up_conv_in_channels= _reverse_down_channels[_block_cnt-1][1], 
                          up_conv_out_channels= _reverse_down_channels[_block_cnt][1],
                                   upsample = _upsample)
        
            self.up_blocks.append(_up_block)
        
        # Build module
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # Finishing layers
        self.out = nn.Conv2d( _reverse_down_channels[-1][1], n_classes, kernel_size = 1, stride = 1)
        self.final = nn.Sigmoid()

    
    def forward(self, x):
        # Create dictionary for layers
        pre_pools = dict()
        
        # Applay input modifications if necessary (pre feature layers)
        _input_layers_count = 0
        if self.input_block != None:
            pre_pools[f"layer_0"] = x
            x = self.input_block(x)
            _input_layers_count += 1
        if self.input_pool != None:
            pre_pools[f"layer_1"] = x
            x = self.input_pool(x)
            _input_layers_count += 1

        # Go trough down blocks
        for i, block in enumerate(self.down_blocks, _input_layers_count):
            x = block(x)
            pre_pools[f"layer_{i}"] = x
        
        # Bridge
        x = self.bridge(x)
                
        # Up blocks
        for i in range(0, len(self.up_blocks)):
           _key = f"layer_{len(pre_pools.keys()) - i-1}"
           x = self.up_blocks[i](x, pre_pools[_key])
        
        # Head part
        x = self.out(x)
        x = self.final(x)
        return x

    def _init_weights(self):
        """
        Method whcih inits random weights to layers.
        """
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }

        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


#*******************************************************#
# Augumentation models
#*******************************************************#  

class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:2],
                input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2,i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t



class LUNA_GRAY(nn.Module):
    """
    Preprocesing for Gray Images
    """
    def __init__(self):
        """
        Define all transofrmations
        """
        super().__init__()

        #self.Add_Chanels = AddChanels()

    def __call__(self, x):
        # Unsqueeze
        #_image = torch.unsqueeze(x, 1)
        _image = (x + 1000.0) / (2000.0)
        # Flip

        # Remove channel
        #_image = _image.squeeze(1)

        # Apply color invert
        _image = _image.repeat(1, 3, 1, 1)
        #_image = self.Add_Chanels(_image)

        return _image


class LUNA_RGB(nn.Module):
    """
    Preprocessing for RGB IMages
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        # Create 3 channel image
        #_image = x.unsqueeze(1)
        _image = (x + 1000.0) / (2000.0)
        _image = _image.repeat(1, 3, 1, 1)
        # Normalize
        _image = fn.normalize(_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Return
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
