import os
import gensim
from Utils_ImageNet.dataloader import *
from Utils_ImageNet.models import *
from Utils_ImageNet.TrainModel import *
from Utils_ImageNet.config import *

# OPTIONAL: Load the "autoreload" extension so that code can change
#%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
#%autoreload 2

# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

# Set root dir
datasetConfig.oversample = True
datasetConfig.gpu = 'cuda:1'
datasetConfig.image_dimension = 512

# Config training
modelConfig.name = 'UNet'
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 2
modelConfig.learning_rate = 1e-4#"Auto" #"Auto" #1e-3
modelConfig.opt_name = 'ADAM'
modelConfig.epochs = 500
modelConfig.gpu = 'cuda:1'
modelConfig.wandb = True
modelConfig.valid_percent = 0.025


#loaderConfig.batch_size = 1
#loaderConfig.number_of_workers = 1

# Set augumentation method
modelConfig.augumentation_model = None #NONE#"GRAY_transform"#"XRAY_transform_GRAY" #None #'XRAY_transform_RGB'
modelConfig.pretrained = False

print(_model_config)
print(_data_config)
print(_loader_config)

datasetConfig.type = 'train'
_train_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'valid'
_valid_data_loader = init_dataloader(_loader_config, _data_config)


_training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, "UNetImageNet/")
_training.freeze_unfreeze_model(freeze = False)
_training.start_training()
