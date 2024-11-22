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
import cv2
import numpy as np
import pandas as pd


def plot_first_image_from_batch(batch, ax):
    _CTs = batch['ct']
    _seriesuids = batch['series_uid']
    _pos_masks = batch['pos']
    _slice_idxs = batch['slice_index']
    elem = dataset.Luna2dSegmentationItem(series_uid=_seriesuids[0], ct=_CTs[0],
                                          pos=_pos_masks[0], slice_index=_slice_idxs[0])
    elem.plot_ct_with_mask(ax=ax)
    ax.set_title(str(elem).replace("---", "\n"))   
    

# Get directories
import os
def list_directories(directory):
    directories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            directories.append(item)
    return directories


# Set up dir paths
LUNA_ROOT = ('LUNADATASET/unzipped')
LUNA_PATHS = LunaPaths(luna_root=LUNA_ROOT)
print('\n',LUNA_PATHS)

# Data precache
NUM_WORKERS = 8
BATCH_SIZE = 32
# how many context slices to use when cropping areas of interest
CONTEXT_SLICES_COUNT = 0

print('Done!')

luna_val_ds = dataset.TrainingLuna2dSegmentationDataset(
    is_val_set=True,
    contextSlices_count=CONTEXT_SLICES_COUNT,
    val_stride=10
)


print('## Creating and iterating val dataloader')
val_dl = DataLoader(luna_val_ds, batch_size=32, num_workers=NUM_WORKERS)

# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Config
_cuda = 'cuda:0'
for _backbone in ['res50', 'res34', 'eff4', 'vgg16']:
        _image_size = get_image_size_for_backbone(backbone=_backbone)
        _lr = 0
        _info = "SCRATCH_"+str(_lr)+"_"+_backbone

        # Model training
        modelConfig.gpu = _cuda
        modelConfig.name = 'UNET'
        modelConfig.backbone = _backbone
        modelConfig.weights_path = None
        modelConfig.loss = 'dice'
        modelConfig.valid_epochs = "1_1"
        modelConfig.early_stopping = 10
        modelConfig.save_epochs = 1
        modelConfig.learning_rate = _lr
        modelConfig.opt_name = 'ADAM'
        modelConfig.epochs = 200
        modelConfig.wandb = False
        modelConfig.custom_info = _info
        modelConfig.image_dimension = _image_size

        # Set augumentation method
        modelConfig.augumentation_model = "LUNA_GRAY"

        # Pretrained
        modelConfig.pretrained = False
        print(_model_config)

        # Replace 'path_to_directory' with the path to the directory you want to list
        _directory = f"RADIOLOGY_NET_LUNA/{_backbone}"

        _list_of_directories = list_directories(_directory)

        # Set keyword
        _keyword = 'SCRATCH'
        #_keyword = 'RadiologyNet'
        #_keyword = 'ImageNet'
        _final_directories_list = []

        # Obtain list of directories
        for _dir in _list_of_directories:
            if _keyword in _dir:
                _final_directories_list.append(_directory+"/"+_dir)


        for _dir in _final_directories_list:
            _models_list = []
            _files = os.listdir(_dir)
            for _item in _files:
               if ".pth" in _item:
                   _models_list.append(_dir+"/"+_item)
            #print(_models_list)
                
            _dir_save_name = _dir.split("/")[-1]
            _dir_save_name = os.path.join('Results', _backbone, _dir_save_name)
            
            if not os.path.exists(_dir_save_name):
                os.makedirs(_dir_save_name)
            else:
                shutil.rmtree(_dir_save_name)           # Removes all the subdirectories!
                os.makedirs(_dir_save_name)
            
            # Go trough files and do predictions
            # Load trening setup
            _training = model_training_app(val_dl, val_dl, _model_config, f"{_dir_save_name}")
            _training.freeze_unfreeze_model(freeze = False)
            
            _final_epoch = []
            _final_iou = []
            _final_dice = []
            for _model in _models_list:
                print("Working with:", _model)
                _training.load_model(_model)
                _training.model_predict_from_dl(val_dl, 'MiddleMan1')
                
                # Work on statistics
                # Obtain files in MiddleMan
                _result_files = os.listdir('MiddleMan1')
                #print(len(_result_files), _result_files)
                # Get images
                
                _name_save = []
                _iou_save = []
                _dice_save = []
                
                 
                for _image_file in _result_files:
                    _true_mask_path = "MiddleMan1"+"/"+_image_file+"/"+"GT_mask.png" 
                    _predicted_mask_path = "MiddleMan1"+"/"+_image_file+"/" + "PRED_mask.png"
                    
                    _gt_mask = cv2.imread(_true_mask_path, cv2.IMREAD_GRAYSCALE)
                    _pred_mask = cv2.imread(_predicted_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Threshold the masks to convert them into binary masks
                    _, _gt_mask = cv2.threshold(_gt_mask, 127, 255, cv2.THRESH_BINARY)
                    _, _pred_mask = cv2.threshold(_pred_mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Calculate intersection and union
                    _intersection = np.logical_and(_gt_mask, _pred_mask)
                    _union = np.logical_or(_gt_mask, _pred_mask)
                    
                    # Calculate IoU (Intersection over Union)
                    _iou = np.sum(_intersection) / np.sum(_union)
                    
                    # Calculate Dice Coefficient
                    _dice_score = 2 * np.sum(_intersection) / (np.sum(_gt_mask)/255 + np.sum(_pred_mask)/255)
                    
                    _name_save.append(_image_file)
                    _iou_save.append(_iou)
                    _dice_save.append(_dice_score)
                df = pd.DataFrame({'Name': _name_save, 'IoU': _iou_save, 'Dice': _dice_save})
                _epoch = _model.split("_")[-1].split(".")[0]
                df.to_excel(_dir_save_name+"/"+str(_epoch)+".xlsx", index=False)
                
                _final_epoch.append(_epoch)
                _final_iou.append(np.mean(_iou_save))
                _final_dice.append(np.mean(_dice_save))
                
            df = pd.DataFrame({'Epoch': _final_epoch, 'IoU': _final_iou, 'Dice': _final_dice})
            df.to_excel(_dir_save_name+"/"+"OVERVIEW.xlsx", index=False)   
            shutil.rmtree('MiddleMan1')
            print(f'I am done with model {_model}')

        print(f"I am done with backbone {_backbone}")



