import torch
import torch.cuda
import glob
import os
from collections import namedtuple
import pandas as pd
import functools
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



# Function for estimation normalization params for RSNA_BONE_DATASET
def estimate_normalization_parameters(dataset_config: cfg.datasetConfig):
    """ 
    Function which find simple statistic for the dataset

    Input:
         Args:
            * dataset_config, see config.py in Utils --> datasetConfig
            All neccessary data for config. Only path to train.csv is required from here

            * save_name, str, name of file where the data is to be saved.
    """

    # Load CSV

    # Obtain csv data
    _df = pd.read_csv(dataset_config.labels_csv_path, index_col='id')
    
    # Go trought data and build data list
    _stats = _df['boneage'].describe()
    
    # Obtain json
    _stats_json = _stats.to_json()
    
    # Save stats
    with open(dataset_config.normalization, 'w') as json_file:
        json_file.write(_stats_json)
        print("HERE")



#**********************************************************************#
# Named tupples for handeling data

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

my_cache = get_cache('Cache/ImageNet_1e-1') #Pure RadiologyNet ImageNet

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
            * sample, dictionary with following attributes: 'image', 'age', and 'gender'
            * image_dimension, integer, scaling factor for images 
        """
        # Obtain extract data
        self.image = sample['image']
        self.age = sample['age']
        self.gender = sample['gender']
        
        # Set dimension
        self.image_dimension = image_dimension

        ## Name
        self.name = sample['image']

        ## Images   
        self.image = cv2.imread(self.image, cv2.IMREAD_GRAYSCALE)
        
        ## Pad image and resize it
        self.__pad_image__(self.image_dimension)

        ## Gender
        if self.gender == 'True':
            self.gender = 1
        else:
            self.gender = 0      
        
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
        return (self.name, self.image, self.gender, self.age)

#**********************************************************************#

# Main class for handeling dataset
class RSNA_dataset:
    """
    Class for handling the dataset. It is the plain train,valid,split

    Random seed is set to 1221

    Link for data:
    https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv
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
        assert os.path.exists(self.dataset_config.imgs_png_home_path), f"Path {self.dataset_config.imgs_png_home_path} does not exist"
        assert os.path.exists(self.dataset_config.labels_csv_path), f"Path {self.dataset_config.labels_csv_path} does not exist"

        # Obtain data lists --> all data as a list with structured paths     
        self.data_list = self.obtain_structured_data()

        # Shuffle it
        random.shuffle(self.data_list)


        # Take partition
        if dataset_config.partition != None:
            self.data_list = self.data_list[0:int(dataset_config.partition * len(self.data_list))]

        # Get count it
        self.samples_cnt = len(self.data_list)

        # Check for label normalization. Parameters are available trough "estimate_normalization_parameters" method
        if self.dataset_config.normalization != None:
            estimate_normalization_parameters(self.dataset_config)
            _df = pd.read_json(self.dataset_config.normalization, typ='series')
            self.min = _df["min"]
            self.max = _df["max"]

    
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

        #Name
        _name = _preprocesed_data[0]

        _image = 0
        # Image and image normalization
        _image = torch.from_numpy(_preprocesed_data[1])
        _image = _image.to(torch.float32)
        _image /= 255.0
       
        # gender
        _gender = _preprocesed_data[2]
        
        # age
        _age = _preprocesed_data[3]

        # Normalize age if necessary to 0-1
        if self.dataset_config.normalization != None:
            _age = (_age - self.min) / (self.max - self.min)

        # Change to tensors
        _age = torch.tensor(_age)
        _age = _age.to(torch.float32)
        _age = _age.unsqueeze(0)
        _gender = torch.tensor(_gender)
        _gender = _gender.to(torch.float32)
        _gender = _gender.unsqueeze(0)

        # Return
        return (_name, _image, _gender, _age)
    


    def obtain_structured_data(self)->dict:
        """
        Function which return a list of data where each data sample is actually a dict with all usefull information
        
        Args:
        
        Output:
            * list where every member is a dict with relevant information: path to image, gender and years.
        
        """

        # Save dir
        _main_output_list = []
        
        # Obtain csv data
        _df = pd.read_csv(self.dataset_config.labels_csv_path, index_col='id')
        #print(_df.head())
        
        # Go trought data and build data list
       
        for _index, _row in _df.iterrows():
            _sample = {}

            _sample['image'] = self.dataset_config.imgs_png_home_path + "/" + str(_index) + ".png"
            _sample['age'] = _row['boneage']
            _sample['gender'] = _row['male']
           
            # Save data into list
            _main_output_list.append(_sample)

        # Return data
        return _main_output_list

       


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
    _ds = RSNA_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl

