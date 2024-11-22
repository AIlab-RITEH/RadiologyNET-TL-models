import os
import gensim
from Utils.dataloader_COVID import *
from Utils.models import *
from Utils.TrainModel import *
from Utils.config import *

# OPTIONAL: Load the "autoreload" extension so that code can change
#%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
#%autoreload 2

# Load configs 
def task(backbone:str):
	_data_config = datasetConfig()
	_loader_config = loaderConfig()
	_model_config = modelConfig()

	datasetConfig.image_dimension = 224
	modelConfig.image_dimension = datasetConfig.image_dimension

	#Create data
	gpu = 'cuda:1'
	loaderConfig.use_gpu = gpu
	loaderConfig.number_of_workers = 4
	loaderConfig.batch_size  = 32
	print(_loader_config)

	# Training dataset
	# Set root dir
	datasetConfig.normal_imgs_png_home_path = "./Data/Normal/images"
	datasetConfig.covid_imgs_png_home_path = "./Data/COVID/images"
	print(_data_config)
	
	datasetConfig.type = 'train'
	_train_data_loader = init_dataloader(_loader_config, _data_config)
	datasetConfig.type = 'valid'
	_valid_data_loader = init_dataloader(_loader_config, _data_config)
	datasetConfig.type = 'test'
	_test_data_loader = init_dataloader(_loader_config, _data_config)


	# Config training -- not
	modelConfig.name = backbone
	modelConfig.valid_epochs = "1_1"
	modelConfig.early_stopping = 10
	modelConfig.learning_rate = 1e-5#"Auto" #"Auto" #1e-3
	modelConfig.opt_name = 'ADAMW'
	modelConfig.epochs = 200
	modelConfig.gpu = gpu
	modelConfig.wandb = False
	modelConfig.valid_percent = None
	modelConfig.number_of_output_classes = 1
	# Set augumentation method
	modelConfig.augumentation_model = "COVID-GRAY"
	modelConfig.pretrained = False
	modelConfig.info = 'Pure-10e-5'

	# Set augumentation method
	modelConfig.augumentation_model = "COVID19_GRAY"
	modelConfig.pretrained = False
	print(_model_config)
		
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
			_TL_MODEL_NAME = [m for m in ['ImageNet', 'RadiologyNet', 'SCRATCH'] if _FILE_PATH.find(m) != -1][0]
			print(f'>> {_FILE_PATH} :: Using {_TL_MODEL_NAME} TL MODEL !!')
			modelConfig.augumentation_model = "COVID19_RGB" if _FILE_PATH.lower().find("imagenet") != -1 else "COVID19_GRAY"
			_training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, os.path.join("Results_Train_Valid_Test", dirpath)+'/')
			_training.load_model(_FILE_PATH)
			_training.model_predict_from_dl(_train_data_loader,"train")
			_training.model_predict_from_dl(_valid_data_loader,"valid")
			_training.model_predict_from_dl(_test_data_loader,"test")

			# also find the xlsx files which contain the training progress
			# it is important to copy THIS after predictions were made, because making predictions
			# deletes everything in the directory where the predictions are saved
			xslx_train_progress = [filename for filename in filenames if filename.endswith('.xlsx')]
			for filename in xslx_train_progress:
				_FILE_PATH = os.path.join(dirpath, filename)
				__new_pth =  os.path.join("Results_Train_Valid_Test", dirpath, f'{filename}_training.xlsx')
				print(f'Found file {filename}. Copying it to {__new_pth}')
				shutil.copy(_FILE_PATH, __new_pth)
				del __new_pth


def main():
    for backbone in [
            'mobileNetV3Large',
            'res18',
        ]:
        print(f'{20*"#"} {backbone.upper()} {20*"#"}')
        task(backbone=backbone)
        print(f'{40*"#"}')

if __name__ == "__main__":
    main()
