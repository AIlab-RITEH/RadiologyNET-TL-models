import os
import gensim
from Utils.dataloader_GRAZPEDWRI import *
from Utils.models import *
from Utils.TrainModel import *
from Utils.config import *

# OPTIONAL: Load the "autoreload" extension so that code can change
#%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
#%autoreload 2

# Load configs 
_data_config = datasetConfig()
_loader_config = loaderConfig()
_model_config = modelConfig()

#Create data
loaderConfig.use_gpu = 'cuda:0'
loaderConfig.number_of_workers = 4
loaderConfig.batch_size  = 32
print(_loader_config)

# Training dataset
datasetConfig.imgs_png_home_path = './Data/Images'
datasetConfig.labels_csv_path = './Data/dataset.csv'
datasetConfig.undersample = True

# Train data loader
datasetConfig.type = 'train'
#datasetConfig.partition = 0.1
_train_dl = init_dataloader(_loader_config, _data_config)

# Valid data loader
datasetConfig.type = 'valid'
_valid_dl = init_dataloader(_loader_config, _data_config)

# Test data loader
datasetConfig.type = 'test'
_test_dl = init_dataloader(_loader_config, _data_config)


datasetConfig.type = 'train'
_train_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'valid'
_valid_data_loader = init_dataloader(_loader_config, _data_config)
datasetConfig.type = 'test'
_test_data_loader = init_dataloader(_loader_config, _data_config)




# Config training
modelConfig.gpu = loaderConfig.use_gpu
modelConfig.valid_epochs = "1_1"
modelConfig.early_stopping = 10
modelConfig.save_epochs = 5
modelConfig.number_of_output_classes = 1
modelConfig.learning_rate = 0
modelConfig.opt_name = 'ADAMW'
modelConfig.epochs = 200
modelConfig.wandb = False
modelConfig.info = "Evaluation"
modelConfig.image_dimension = 224

# Set augumentation method
modelConfig.augumentation_model = "GRAZPEDWRI_RGB"
modelConfig.pretrained = False
print(_model_config)

for backbone in [
		'dense121',
    	'res34',
]:
	modelConfig.name = backbone
	MODELS_DIR = os.path.join("MODELS", backbone)
	for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
		best_model_file = [filename for filename in filenames if filename.endswith('best_model.pth')]
		if best_model_file.__len__() != 1:
			print(f'Either there were multiple best_model files, or none were found. Skipping "{best_model_file}"')
			continue
		else:
			# there should only be one file
			best_model_file = best_model_file[0]
			_FILE_PATH = os.path.join(dirpath, best_model_file)
			_TL_MODEL_NAME = [m for m in ['imagenet', 'radiologynet', 'scratch'] if _FILE_PATH.lower().find(m) != -1][0]
			print(f'>> {_FILE_PATH} :: Using {_TL_MODEL_NAME} TL MODEL !!')
			
			# use RGB aug model only if imagenet!! (that's because of imagenet 3-channel scaling)
			modelConfig.augumentation_model = "GRAZPEDWRI_RGB" if _FILE_PATH.lower().find("imagenet") != -1 else "GRAZPEDWRI_GRAY"
			assert _FILE_PATH.find(modelConfig.name) != -1, f"Config set to model {modelConfig.name}, but model path set to {_FILE_PATH}!"
			_training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, os.path.join("Results_Train_Valid_Test", dirpath)+'/')
			_training.load_model(_FILE_PATH)
			_training.model_predict_from_dl(_train_data_loader,"train")
			_training.model_predict_from_dl(_valid_data_loader,"valid")
			_training.model_predict_from_dl(_test_data_loader,"test")

			# also find the xlsx files which contain the training progress
			xslx_train_progress = [filename for filename in filenames if filename.endswith('.xlsx')]
			for filename in xslx_train_progress:
				_FILE_PATH = os.path.join(dirpath, filename)
				__new_pth =  os.path.join("Results_Train_Valid_Test", dirpath, f'{filename}_training.xlsx')
				print(f'Found file {filename}. Copying it to {__new_pth}')
				shutil.copy(_FILE_PATH, __new_pth)
				del __new_pth

