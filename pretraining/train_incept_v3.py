import os
import gensim
from Utils.dataloader import *
from Utils.models import *
from Utils.TrainModel import *
from Utils.config import *



# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Set root dir
datasetConfig.oversample = True
datasetConfig.gpu = 'cuda:3'
datasetConfig.blacklist = [16,32,35,39, 0, 15, 17, 19, 21, 22, 23, 31, 36, 48]
datasetConfig.imgs_png_home_path = 'RadiologyNetData4TransferLeraning/converted_images224x224'
datasetConfig.labels_csv_path = 'RadiologyNetData4TransferLeraning/RadiologyNET_LABELS.csv'
datasetConfig.image_dimension = 299

# Config training
modelConfig.name = 'incept_v3'
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 10
modelConfig.learning_rate = 1e-3#"Auto" #"Auto" #1e-3
modelConfig.opt_name = 'ADAMW'
modelConfig.epochs = 500
modelConfig.gpu = 'cuda:3'
modelConfig.wandb = False
modelConfig.valid_percent = 0.05
modelConfig.number_of_output_classes = 50 - len(datasetConfig.blacklist)

#loaderConfig.batch_size = 1
#loaderConfig.number_of_workers = 1

# Set augumentation method
modelConfig.augumentation_model = "XRAY_transform_GRAY"#"GRAY_transform"#"XRAY_transform_GRAY" #None #'XRAY_transform_RGB'
modelConfig.pretrained = False

print(_model_config)
print(_data_config)
print(_loader_config)

datasetConfig.type = 'train'
_train_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'valid'
_valid_data_loader = init_dataloader(_loader_config, _data_config)

print(len(_valid_data_loader), len(_train_data_loader))

_training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, "incept_v3/")
_training.freeze_unfreeze_model(freeze = False)
_training.start_training()
