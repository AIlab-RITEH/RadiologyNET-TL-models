from luna.paths import LunaPaths
from luna import candidate_info, dataset
from luna.ct import Ct
from luna.prepcache import LunaPrepCacheApp
from torch.utils.data import DataLoader

from Utils.models import *
from Utils.TrainModelLUNA import *
from Utils.config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def list_directories(directory):
    directories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            directories.append(item)
    return directories

def filter_dirs(list_of_directories, keyword):
    _final_directories_list = []
    for _dir in list_of_directories:
        if keyword in _dir:
            _final_directories_list.append(_directory+"/"+_dir)
    return _final_directories_list

def obtain_best_model(_dirs):
    _name_list = []
    _epoch_list = []
    _iou_list = []
    _dice_list = []
    for _dir in _dirs:
        _file_path = _dir+"/"+"OVERVIEW.xlsx"
        _df = pd.read_excel(_file_path)
        _max_row = _df['IoU'].idxmax()
        _max_row = _df.iloc[_max_row]
        _name_list.append(_dir.split("/")[-1])
        _epoch_list.append(_max_row['Epoch'])
        _iou_list.append(_max_row['IoU'])
        _dice_list.append(_max_row['Dice'])
    
    _df = pd.DataFrame({'Name': _name_list, 'Epoch': _epoch_list, 'IoU': _iou_list, 'Dice': _dice_list})
    _df = _df.sort_values(by='IoU', ascending=False)
    print(_df)
        

# Replace 'path_to_directory' with the path to the directory you want to list
_directory = "RADIOLOGY_NET_LUNA/INFERENCE-EVALUATION/eff4"

print(f"WOKRING ON {_directory}")

_list_of_directories = list_directories(_directory)
_pure_dirs_list = filter_dirs(_list_of_directories, 'SCRATCH')
_imagenet_dirs_list = filter_dirs(_list_of_directories, 'ImageNet')
_radiologynet_dirs_list = filter_dirs(_list_of_directories, 'Radiology')

_best = obtain_best_model(_pure_dirs_list)
_best = obtain_best_model(_imagenet_dirs_list)
_best = obtain_best_model(_radiologynet_dirs_list)

print("##############################################################")

# Replace 'path_to_directory' with the path to the directory you want to list
_directory = "RADIOLOGY_NET_LUNA/INFERENCE-EVALUATION/res34"
print(f"WOKRING ON {_directory}")
_list_of_directories = list_directories(_directory)
print(_list_of_directories)
_pure_dirs_list = filter_dirs(_list_of_directories, 'SCRATCH')
_imagenet_dirs_list = filter_dirs(_list_of_directories, 'ImageNet')
_radiologynet_dirs_list = filter_dirs(_list_of_directories, 'Radiology')

_best = obtain_best_model(_pure_dirs_list)
_best = obtain_best_model(_imagenet_dirs_list)
_best = obtain_best_model(_radiologynet_dirs_list)
print("##############################################################")


# Replace 'path_to_directory' with the path to the directory you want to list
_directory = "RADIOLOGY_NET_LUNA/INFERENCE-EVALUATION/vgg16"
print(f"WOKRING ON {_directory}")
_list_of_directories = list_directories(_directory)
_pure_dirs_list = filter_dirs(_list_of_directories, 'SCRATCH')
_imagenet_dirs_list = filter_dirs(_list_of_directories, 'ImageNet')
_radiologynet_dirs_list = filter_dirs(_list_of_directories, 'Radiology')

_best = obtain_best_model(_pure_dirs_list)
_best = obtain_best_model(_imagenet_dirs_list)
_best = obtain_best_model(_radiologynet_dirs_list)
print("##############################################################")


# Replace 'path_to_directory' with the path to the directory you want to list
_directory = "RADIOLOGY_NET_LUNA/INFERENCE-EVALUATION/res50"
print(f"WOKRING ON {_directory}")
_list_of_directories = list_directories(_directory)
_pure_dirs_list = filter_dirs(_list_of_directories, 'SCRATCH')
_imagenet_dirs_list = filter_dirs(_list_of_directories, 'ImageNet')
_radiologynet_dirs_list = filter_dirs(_list_of_directories, 'Radiology')

_best = obtain_best_model(_pure_dirs_list)
_best = obtain_best_model(_imagenet_dirs_list)

# Replace 'path_to_directory' with the path to the directory you want to list
_directory = "RADIOLOGY_NET_LUNA/INFERENCE-EVALUATION/res50-2"
print(f"WOKRING ON {_directory}")
_list_of_directories = list_directories(_directory)
_pure_dirs_list = filter_dirs(_list_of_directories, 'SCRATCH')
_imagenet_dirs_list = filter_dirs(_list_of_directories, 'ImageNet')
_radiologynet_dirs_list = filter_dirs(_list_of_directories, 'Radiology')

_best = obtain_best_model(_pure_dirs_list)
_best = obtain_best_model(_imagenet_dirs_list)
_best = obtain_best_model(_radiologynet_dirs_list)
print("##############################################################")

