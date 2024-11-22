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
import Utils_ImageNet.config as cfg


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

my_cache = get_cache('Cache_ImageNet')

@my_cache.memoize(typed=True)
def get_data_sample(image_path, image_dimension):
    """
    Middleman function for cashing Fast is smooth, smooth is fast
    """
    _data = ProcessData(image_path, image_dimension)
    _output = _data.get_sample()
    return _output
#**********************************************************************#
class ProcessData: 
    """
    Class for loading data from json
    """
    def __init__(self, image_path, image_dimension):
        """
        Init function.

        Args: 
            * sample, dictionary with following attributes: 'id' , 'examID' , 'modality' ,'bpe', 'label', 'image_path'

            * dimensions, integer, scaling factor for images 
        """
        # Obtain extract data

        ## Images   
        _image_path = image_path
        self.image = cv2.imread(_image_path, cv2.IMREAD_COLOR)
        
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
        # Get the dimensions of the image
        _height, _width, _ = self.image.shape

        # Determine the size of the largest square that fits within the image dimensions
        _side = min(_height, _width)

        # Calculate the coordinates for cropping the largest square
        _start_x = (_width - _side) // 2
        _start_y = (_height - _side) // 2
        _end_x = _start_x + _side
        _end_y = _start_y + _side

        # Crop the largest square from the image
        _cropped_square = self.image[_start_y:_end_y, _start_x:_end_x]

        # Resize the cropped square to 512x512 using OpenCV
        self.image = cv2.resize(_cropped_square, (image_desired_size, image_desired_size))
        
        # Change channel order for performance 
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        
    def get_sample(self):
        """
        Return sample --> loaded image and its annotations
        """
        return (self.image, 'dummy', 'dummy','dummy', 'dummy', 'dummy')
        

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
        
        # Select dataset and store it in self.data 
        selector = lambda type: {
        'train': self.get_image_paths(self.dataset_config.train_images_path),
        'valid': self.get_image_paths(self.dataset_config.valid_images_path),
        'test': self.get_image_paths(self.dataset_config.test_images_path),
        }[type]        

        self.data_list = selector(self.dataset_config.type)
        random.shuffle(self.data_list)
        # Get count it
        self.samples_cnt = len(self.data_list)

    def get_image_paths(self, root_folder):
        _image_paths = []
        for _root, _dirs, _files in os.walk(root_folder):
            for _file in _files:
                if _file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    _image_paths.append(os.path.join(_root, _file))
        return _image_paths

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
        _image_path = self.data_list[indx]

        # Obtain image (input) and annotation(output)
        _preprocesed_data = get_data_sample(_image_path, self.dataset_config.image_dimension)     

        # Image and image normalization
        _image = torch.from_numpy(_preprocesed_data[0])
        _image = _image.to(torch.float32)
        _image /= 255.0
        _image = _image.permute(2, 0, 1)
       
        # id
        _id = 'dummy'

        # exam ID
        _examID = 'dummy'

        # modality
        _modality = 'dummy'

        # bpe
        _bpe = 'dummy'

        # label
        _label = 'dummy'

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

