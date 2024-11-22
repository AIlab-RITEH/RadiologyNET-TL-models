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
    Caching Descriptor function
    """
    return diskcache.FanoutCache('data_cache/' + scope_str,
                       shards=20,
                       timeout=1,
                       size_limit=3e11,
                    )

my_cache = get_cache('BrainTumor_Scratch')

@my_cache.memoize(typed=True)
def get_data_sample(sample, image_dimension):
    """
    Middleman function for caching Fast is smooth, smooth is fast
    """
    _data = ProcessData(sample, image_dimension)
    _output = _data.get_sample()
    return _output


#**********************************************************************#
class ProcessData: 
    """
    Class for loading data
    """
    def __init__(self, sample, image_dimension):
        """
        Init function.

        Args: 
            * sample

            * dimensions, integer, scaling factor for images 
        """
        # Obtain extract data
        _image_path = sample
        self.label = sample.split(os.sep)[-2]
        self.string_label = self.label
        self.path = sample
        _all_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

        self.label = _all_labels.index(self.label)

        ## Images   
        self.image = cv2.imread(_image_path, cv2.IMREAD_GRAYSCALE)
        
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
        return (self.image, self.label, self.string_label, self.path)
        

#**********************************************************************#

# Main class for handeling dataset
class BrainTumor_dataset:
    """
    Class for handling the dataset. It is the plain train,valid,split

    Random seed is set to 1221
    """

    def __init__(self, dataset_config: cfg.DatasetConfig):
        """
        Init function which handles the dataset loading
        Args:
            * dataset_config, see config.py in Utils --> DatasetConfig
            All neccessary data for config
                
        """

        # Set self config
        self.dataset_config = dataset_config

        # Set random seed
        random.seed(1221)
        
    
        # Check if data path is ok
        assert os.path.exists(self.dataset_config.train_imgs_home_path), f"Path {self.dataset_config.train_imgs_home_path} does not exist"
        assert os.path.exists(self.dataset_config.test_imgs_home_path), f"Path {self.dataset_config.train_imgs_home_path} does not exist"

        # Obtain data lists --> all data as json
        _data_list_train = self.obtain_structured_data(self.dataset_config.train_imgs_home_path)
        _data_list_test = self.obtain_structured_data(self.dataset_config.test_imgs_home_path)
        
        # Create splits
        _data_list_splitted = self.split_to_train_val_test(_data_list_train, split_ratio = self.dataset_config.split_ratio)   
        _data_list_test = self.obtain_test_data(_data_list_test)

        # Select dataset and store it in self.data 
        selector = lambda type, data_list_splitted: {
        'train': data_list_splitted[0],
        'valid': data_list_splitted[1],
        'test': _data_list_test,
        }[type]        

        self.data_list = selector(self.dataset_config.type, _data_list_splitted)

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

        # full label 
        _label_string = _preprocesed_data[2]

        # path
        _path = _preprocesed_data[3]

        # Return
        return (_image, _label, _label_string, _path)


    def obtain_structured_data(self, data_path)->dict:
        """
        Function which list files and builds up dict with all relevent data
        
        Args:
            * data_path, string, path to root dir
        
        Output:
            * dict where keys are clusters labels and keys are lists of samples belonging to that label.
            Each sample has path to image, modality, examid and body part examined.
        
        """   

        # Save dir
        _main_output_dir = {}
        
        # Obtain all labels
        _labels = os.listdir(data_path)
        
        # Go trough labels and build dataset
        for _label in _labels:
            _dir_path = os.path.join(data_path, _label)

            # List all saples for a label
            _samples = os.listdir(_dir_path)

            # Save found samples
            for _sample in _samples:
                _full_sample_path = os.path.join(_dir_path, _sample)
                # Save data into dir
                if _label in _main_output_dir:
                    _main_output_dir[_label].append(_full_sample_path)
                else:
                    _main_output_dir[_label] = [_full_sample_path]
        
        # Return data
        return _main_output_dir
    

    def obtain_test_data(self, data_dict:dict)->list:
        """
        Function which obrains test dataset as a list
        """
         # Set random seed
        random.seed(1221)
        
        # Storage
        _test_data_list = []

        # Fill up list
        for _key in data_dict.keys():
            _test_data_list = _test_data_list + data_dict[_key]

        # Random shuffle
        random.shuffle(_test_data_list)

        return _test_data_list

    def split_to_train_val_test(self, data_dict:dict, split_ratio :float = 0.75)->list:
        """
        Function which accepts parsed dict by filtered out data and obtained structured path
        
        Args:
            * data_dict, dictionary, dictionary which containes out of fractures
            * train_split_ratio, float, split ratio for dataset. valid=test=1-_train_split_ratio/2
        
        Output:
            * three lists of samples: train, valid, test.
        """
        # Set random seed
        random.seed(1221)
        
        # Set split ratio # Valid/test is what is left from
        _train_split_ratio = split_ratio
        
        
        # Define storage
        _train_data_list = []
        _validation_data_list = []
        
        _cnt = 0

                
        # Create train validation and test sets
        for _key in data_dict.keys():
            # Grab data and shuffle it

            _size = len(data_dict[_key])
            random.shuffle(data_dict[_key])
            _sample_list = data_dict[_key]

            # Split it
            _train_split = _sample_list[: int(_train_split_ratio*(_size))] 
            _valid_split = _sample_list[int(_train_split_ratio*(_size)):]
            
        
            # Save them 
            _cnt += len(_train_split)
            _train_data_list = _train_data_list + _train_split
            _validation_data_list = _validation_data_list + _valid_split
        
        # Return values
        random.shuffle(_train_data_list)
        random.shuffle(_validation_data_list)
        return([_train_data_list, _validation_data_list])
    


# Generate dataloader
def init_dataloader(data_loader_params: cfg.LoaderConfig, dataset_params: cfg.DatasetConfig)->DataLoader:
    """
        Init of the  data loader.
        Creating wrapper arround data class. 

        ARGS:
            * batch_size, int, size of the batch
            * num_wokers, int, number of workers for data loading 
            * use_gpu, boolean, if gpu used
            * dataset_info --> data_params object

        Output:
            * Torch DataLoader
    """
    _ds = BrainTumor_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl

