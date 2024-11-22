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

import radiologynet.tools.raw.io as raw_io
import radiologynet.tools.raw.utils as raw_utils
import radiologynet.tools.visualization.image as plt_image
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

my_cache = get_cache('eff6_r')#vit_b_32-3

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
            * sample, dictionary with following attributes: 'id' , 'examID' , 'modality' ,'bpe', 'label', 'image_path'

            * dimensions, integer, scaling factor for images 
        """
        # Obtain extract data
        self.id = sample['id']
        self.examID = sample['examID']
        self.modality = sample['modality']
        self.bpe = sample['bpe']
        self.label = sample['label']
        
        ## Images   
        _image_path = sample['image_path']
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
        return (self.image, self.label, self.id, self.examID, self.modality, self.bpe)
        

#**********************************************************************#

# Main class for handeling dataset
class Radiology_net_dataset:
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
        assert os.path.exists(self.dataset_config.imgs_png_home_path), f"Path {self.dataset_config.imgs_png_home_path} does not exist"
        assert os.path.exists(self.dataset_config.labels_csv_path), f"Path {self.dataset_config.labels_csv_path} does not exist"

        # Obtain data lists --> all data as json
        _data_list = self.obtain_structured_data()

        # Filter out data based on number of items in dataset
        _filtered_data_list = self.filter_out_data(_data_list, 
                                                  self.dataset_config.threshold)

        # Create splits
        _data_list_splitted = self.split_to_train_val_test(_filtered_data_list, 
                                                                            split_ratio = self.dataset_config.split_ratio,
                                                                            oversample = self.dataset_config.oversample,
                                                                            verbose = self.dataset_config.verbose)   

        # Select dataset and store it in self.data 
        selector = lambda type, data_list_splitted: {
        'train': data_list_splitted[0],
        'valid': data_list_splitted[1],
        'test': data_list_splitted[2],
        }[type]
        

		
        self.data_list = selector(self.dataset_config.type, _data_list_splitted)

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
       
        # id
        _id = _preprocesed_data[2]

        # exam ID
        _examID = _preprocesed_data[3]

        # modality
        _modality = _preprocesed_data[4]

        # bpe
        _bpe = _preprocesed_data[5]

        # label
        _label = torch.tensor(_preprocesed_data[1])

        # Return
        return (_image, _label, _id, _examID, _modality, _bpe)

    def obtain_structured_data(self)->dict:
        """
        Function which list files and builds up dict with all relevent data
        
        Args:
        
        Output:
            * dict where keys are clusters labels and keys are lists of samples belonging to that label.
            Each sample has path to image, modality, examid and body part examined.
        
        """

        # Save dir
        _main_output_dir = {}
        
        # Obtain csv data
        _df = pd.read_csv(self.dataset_config.labels_csv_path, index_col='id')

        # Create label remaping if blacklist is on if we delete 38, then 39 is 38
        if self.dataset_config.blacklist != []:
            _current_label = 0
            _label_remap_dict = {}
            for _i in range(self.dataset_config.number_of_labels):
                if _i in self.dataset_config.blacklist:
                    continue
                else:
                    _label_remap_dict[_i] = _current_label
                    _current_label += 1


        # Go trought data and build data list
        # This is for testing purposes to load only part of the dataset
        #STOP_CRITERIA = 100000
        #CNT = 0
        for _index, _row in _df.iterrows():
            #if CNT == STOP_CRITERIA:
            #    break
            #CNT += 1
            _sample = {}
            _sample['id'] = _index
            _sample['examID'] = _row['ExamID']
            _sample['modality'] = _row['Modality']
            _sample['bpe'] = _row['BodyPartExamined']
            _sample['label'] = _row['Cluster']
            
            # Check for blacklist
            if _sample['label'] in self.dataset_config.blacklist:
                continue
            
            # Applay remap if blacklist is on 
            if self.dataset_config.blacklist != []:
                _sample['label'] = _label_remap_dict[_sample['label']]

            # Get image path
            _image_path = raw_io.get_path_from_id(
                    disk_label=self.dataset_config.imgs_png_home_path,
                    id=_index,
                    extension='png',
                    modality= _row['Modality']
                    )
            _image_path = os.path.join(_image_path, '0.png')
            _sample['image_path'] = _image_path
        
            # Save data into dir
            if _sample['label'] in _main_output_dir:
                _main_output_dir[_sample['label']].append(_sample)
            else:
                _main_output_dir[_sample['label']] = [_sample]

        # Return data
        return _main_output_dir

    def filter_out_data(self, input_dict:dict, minumum_frequency:int = 0)->dict:
        """
        Function which filter out only data with 0 or more occurances
        
        Args:
            * Dictionary obtained by the function: obtain_structured_data
            * minimum frequency: Minimum number of occurances of file
        
        Output: filtered dict
        """
        # Sort dictionary
        _input_dict = OrderedDict(sorted(input_dict.items(), key=lambda x: len(x[1]), reverse=True))
        
        # Create output dir
        _main_output_dir = _input_dict.copy()
        
        # Filter
        for _item in _input_dict:
            if len(_input_dict[_item]) < minumum_frequency:
                del _main_output_dir[_item] 
        
        # Export
        return _main_output_dir    

    def split_to_train_val_test(self, data_dict:dict, split_ratio :float = 0.75, oversample = False, verbose : bool = 0)->list:
        """
        Function which accepts parsed dict by filtered out data and obtained structured path
        
        Args:
            * data_dict, dictionary, dictionary which containes out of fractures
            * train_split_ratio, float, split ratio for dataset. valid=test=1-_train_split_ratio/2
            * ovesample, bool, oversample training set so each class has same number of instances (if true)
            * verbose, boolean, print statistics on the run
        
        Output:
            * three lists of samples: train, valid, test.
        """
        # Set random seed
        random.seed(1221)
        
        # Set split ratio # Valid/test is what is left from
        _train_split_ratio = split_ratio
        _valid_split_ratio = (1.0 - split_ratio) / 2.0  
        
        # Define storage
        _train_data_list = []
        _validation_data_list = []
        _test_data_list = []
        
        _cnt = 0

        # Print header
        if verbose:
            print("Data statistics")
            print(f"{'Name':<10} | {'Total':<10} | {'Train':<10} | {'Valid':<10} | {'Test':<10}")
       
        # Find bigest number of smaples for oversampling
        if oversample:
            _max_number_of_samples = int(len(max(data_dict.values(), key=len)) * _train_split_ratio)
                
        # Create train validation and test sets
        for _key in data_dict.keys():
            # Grab data and shuffle it

            _size = len(data_dict[_key])
            random.shuffle(data_dict[_key])
            _sample_list = data_dict[_key]

            # Split it
            _train_split = _sample_list[: int(_train_split_ratio*(_size))] 
            _valid_split = _sample_list[int(_train_split_ratio*(_size)): int((_train_split_ratio+_valid_split_ratio)*(_size))]
            _test_split = _sample_list[int((_train_split_ratio+_valid_split_ratio)*(_size)) : ]
            
            # Oversample if necessary
            if oversample:
                _sampling_size = len(_train_split)
                _multiply_size = int(_max_number_of_samples / _sampling_size)
                _residue_size = _max_number_of_samples -_sampling_size * _multiply_size

                _train_split_m = _train_split * _multiply_size
                _train_split = _train_split_m + _train_split[0:_residue_size]

            # Verbose
            if verbose:
                print(f"{_key:<10} | {_size:<10} | {len(_train_split):<10} | {len(_valid_split):<10} | {len(_test_split):<10}")
        
            # Save them 
            _cnt += len(_train_split)
            _train_data_list = _train_data_list + _train_split
            _validation_data_list = _validation_data_list + _valid_split
            _test_data_list = _test_data_list + _test_split
        
        # Final statictics
        # Verbose
        if verbose:
            print(f"Oversample: {oversample}")
            print(f"{'Dataset':<10} | {sum([len(_train_data_list), len(_validation_data_list), len(_test_data_list)]):<10} | {len(_train_data_list):<10} | {len(_validation_data_list):<10} | {len(_test_data_list):<10}")

        # Return values
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
    _ds = Radiology_net_dataset(dataset_params)

    _dl = DataLoader(
        _ds,
        batch_size = data_loader_params.batch_size,
        num_workers = data_loader_params.number_of_workers,
        pin_memory = data_loader_params.use_gpu,
    )  
    return _dl

