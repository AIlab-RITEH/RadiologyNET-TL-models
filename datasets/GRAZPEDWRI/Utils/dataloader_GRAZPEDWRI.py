import torch
import torch.cuda
import glob
import os
from collections import namedtuple
import pandas as pd
import json
from collections import *
import random
import diskcache
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torchvision.transforms.functional as fn

import Utils.config as cfg


#**********************************************************************#
# Named tupples for handeling data

"""
Normalization_params:
px_area_scaler --> normalization for scaling
original_width_scaler --> scaler for width
original_height_scaler --> scaler for hegiht
image_scaler --> dimension of the image
"""
normalization_params = namedtuple(
    'normalization_params',
    'px_area_scaler, original_width_scaler, original_height_scaler, image_scaler',
)


#**********************************************************************#
# Data loading helpful functions

def get_cache(scope_str):
    """
    Cashing Descriptor function
    """
    return diskcache.FanoutCache('data_cache/' + scope_str,
                       shards=20,
                       timeout=1,
                       size_limit=3e11,
                       )

my_cache = get_cache('BrainTumor')

@my_cache.memoize(typed=True)
def get_data_sample(sample, image_dimension):
    """
    Middleman function for cashing Fast is smooth, smooth is fast
    """
    _data = ProcessData(sample, image_dimension)
    _output = _data.get_sample()
    return _output


#**********************************************************************#
class ProcessData: 
    """
    Class for loading data from json
    """
    def __init__(self, sample, image_dimension):
        """
        Init function.

        Args: 
            * sample, dictionary with following attributes: 'path' , 'exam' , 'z' ,'series_no', 'label'

            * dimensions, integer, scaling factor for images 
        """
        self.path = sample['path']
        self.label = int(sample['label'])

        ## Images   
        self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        
        # Set dimension
        self.image_dimension = image_dimension
        
        # Pad image and resize it
        self.__pad_image__(self.image_dimension)
        
    def __pad_image__(self, image_desired_size:int):
        """
        Script for resizing the image to the desired dimensions
        First the image is resized then it is zero-padded to the desired 
        size given in the argument

        Args:
            * image_desired_dimension, int, new size of the image

        Output:
            * None, self.image is updated
        """
        # Grab the old_size
        _old_size = self.image.shape[:2] # old_size is in (height, width) format
        # Calculate new size
        _ratio = float(image_desired_size)/max(_old_size)
        _new_size = tuple([int(_x*_ratio) for _x in _old_size])

        # new_size should be in (width, height) format
        self.image = cv2.resize(self.image, (_new_size[1], _new_size[0]))
        
        
        # Calculate padding
        _delta_w = image_desired_size - _new_size[1]
        _delta_h = image_desired_size - _new_size[0]
        _top, _bottom = _delta_h//2, _delta_h-(_delta_h//2)
        _left, _right = _delta_w//2, _delta_w-(_delta_w//2)
        
        # Pad
        color = [0]
        
        self.image = cv2.copyMakeBorder(self.image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=color)
        # Change to grayscale
        
        self.image = self.image

        
    def get_sample(self):
        """
        Return sample --> loaded image and its annotations
        """
        return (self.image, self.label)
        

#**********************************************************************#

# Main class for handeling dataset
class GrazPedWri_dataset:
    """
    Class for handling the dataset. It is the plain train,valid,split

    Random seed is set to 1221
    """

    def __init__(self, dataset_config: cfg.datasetConfig):
        """
        Init function which handles the dataset loading
        Args:
            * dataset_config, see config.py in Utils --> datasetConfig
            All neccessary data for config
                
        """

        # Set self config
        self.dataset_config = dataset_config

        # Set random seed
        random.seed(1221)
        
    
        # Check if data path is ok
        assert os.path.exists(self.dataset_config.labels_csv_path), f"Path {self.dataset_config.data_path} does not exist"

        # Obtain data lists --> all data as json
        _structured_data_dict = self.obtain_structured_data(self.dataset_config.labels_csv_path, self.dataset_config.imgs_png_home_path)
        
        # Create splits
        _train_data_list, _valid_data_list, _test_data_list = self.split_into_sets(_structured_data_dict,
                                                                                    undersample = self.dataset_config.undersample,
                                                                                    split_ratio = self.dataset_config.split_ratio)   
                                                                            

        # Select dataset and store it in self.data 
        selector = lambda type: {
        'train': _train_data_list,
        'valid': _valid_data_list,
        'test': _test_data_list,
        }[type]        

        self.data_list = selector(self.dataset_config.type)

        # Take part of the data
        if self.dataset_config.type == 'train' and self.dataset_config.partition != None:
            self.data_list = self.data_list[0:int(dataset_config.partition * len(self.data_list))]
            
        # Get count it 
        self.samples_cnt = len(self.data_list)

    
    def __len__(self):
        """
        Returns number of samples in dataset
        """
        return self.samples_cnt
    
    def shuffle_samples(self):
        """
        Simply shuffles the dataset -- necessary for batches
        """
        # Shuffeling dataset
        random.seed(1221)
        random.shuffle(self.data_list)


    def __getitem__(self, indx):
        """
        Gets data from the dataset

        Args:
            * indx, int, index of the data sample
        
        Output: data sample
        """
        # Get sample
        _sample = self.data_list[indx]

        # Obtain image (input) and annotation(output)
        _preprocesed_data = get_data_sample(_sample, self.dataset_config.image_dimension)     
        # Image and image normalization
        _image = torch.from_numpy(_preprocesed_data[0])
        _image = _image.to(torch.float32)
        _image /= 255.0

        # label

        _label = torch.tensor(_preprocesed_data[1])
        _label = _label.float()
        _label = _label.unsqueeze(0)

        # Return
        return (_image, _label)
  

    def obtain_structured_data(self, data_path, image_path)->dict:
        """
        Function which list files and builds up list with all relevent data
        
        Args:
            * data_path, string, path to root dir
            * image_path, string, path to dir where images are storaged
        
        Output:
            * list containing sturcutred data for each data sample: path, examid, z, series_no 
            and label
        
        """   
        # Input path
        _input_path = data_path

        # Load data
        _df = pd.read_csv(data_path)
    
        # Save list
        _data_list = []
    
        # Go through labels and build a list
        for _i, _row in _df.iterrows():
            # Obtain data of focus
            _file = _row['filestem']
            _osteopenia = _row['osteopenia']

            # Obtain label
            if np.isnan(_osteopenia):
                _label = 0
            else:
                _label = 1

            # Save
            _file = _file + ".png"
            _item = {'path': os.path.join(image_path,_file), 
                     'label': _label, 
                    }
            _data_list.append(_item)

        # Sort it in data dict
        _data_dict = {}
        for _sample in _data_list:
            if _sample['label'] in _data_dict:
                _data_dict[_sample['label']].append(_sample)
            else:
                _data_dict[_sample['label']] = [_sample]
    
        # Return data
        return _data_dict

    def split_into_sets(self, data_dict:dict, undersample: bool= False, split_ratio :float = 0.75)->list:
        """
        Function which accepts parsed dict by filtered out data and obtained structured path
        
        Args:
            * data_dict, dict, dict which containes out of fractures
            * undersample, bool, to undersample classes in training set
            * train_split_ratio, float, split ratio for dataset. valid=test=1-_train_split_ratio/2
        
        Output:
            * three lists of samples: train, valid, test.
        """
        # Set random seed
        random.seed(1221)
        
        # Set split ratio # Valid/test is what is left from
        _train_split_ratio = split_ratio

        if undersample:
            _min_number_of_samples = int(len(min(data_dict.values(), key=len)))
            for _key in data_dict.keys():
                data_dict[_key] = data_dict[_key][0:_min_number_of_samples]
    
        # Dict for required number of instances
        _required_number_of_instances_dict = {}
        for _key in data_dict:
            _required_number_of_instances_dict[_key] = int(len(data_dict[_key])* (1-split_ratio) / 2)

        # Define storage
        _train_data_list = []
        _validation_data_list = []
        _test_data_list = []

        # Populate 
        for _key in data_dict.keys():
            # Grab data and shuffle it

            _size = len(data_dict[_key])
            random.shuffle(data_dict[_key])
            _sample_list = data_dict[_key]
                
            # Split it
            _train_split = _sample_list[: int(_train_split_ratio*(_size))] 
            _valid_split = _sample_list[int(_train_split_ratio*(_size)):int(_train_split_ratio*(_size)+_required_number_of_instances_dict[_key])]
            _test_split = _sample_list[int(_train_split_ratio*(_size)+_required_number_of_instances_dict[_key]):]
            
        
            # Save them 
            _train_data_list = _train_data_list + _train_split
            _validation_data_list = _validation_data_list + _valid_split
            _test_data_list = _test_data_list + _test_split
       

        # Populate the training_data_list
        
        random.shuffle(_train_data_list)
        random.shuffle(_validation_data_list)
        random.shuffle(_test_data_list)
        return([_train_data_list, _validation_data_list, _test_data_list])
    


# Generate dataloader
def init_dataloader(data_loader_params: cfg.loaderConfig, dataset_params: cfg.datasetConfig)->DataLoader:
    """
        Init of the  data loader. NOT TESTED FOR MULTIPLE GPU
        Creating wrapper arround data class. 

        ARGS:
            * batch_size, int, size of the batch
            * num_wokers, int, number of workers for data loading 
            * use_gpu, boolean, if gpu used
            * dataset_info --> data_params object

        Output:
            * Torch DataLoader
    """
    _ds = GrazPedWri_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl
