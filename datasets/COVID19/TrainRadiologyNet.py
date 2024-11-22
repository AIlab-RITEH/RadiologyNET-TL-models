from Utils.dataloader_COVID import *
from Utils.config import *
from Utils.models import *
from Utils.TrainModel import *
from multiprocessing import Process

def task(lr, data_partition:float=None, run_idx:int=None, backbone: str='res152'):
	_cuda = 'cuda:0'
	_image_size = 224

	_lr = lr
	_info = "RadiologyNet_"+str(lr)+f"_{backbone}"
	_info += f'_partition-{data_partition}' if data_partition is not None else ''
	_info += f'_run-{run_idx:03}' if run_idx is not None else ''
	print(_info)
	
	# Load configs 
	_data_config = datasetConfig()
	_loader_config = loaderConfig()
	_model_config = modelConfig()

	loaderConfig.use_gpu = _cuda
	loaderConfig.number_of_workers = 4
	loaderConfig.batch_size  = 32
	print(_loader_config)
	datasetConfig.partition = _partition

	# Training dataset
	datasetConfig.gpu = _cuda
	datasetConfig.image_dimension = 224
	datasetConfig.normal_imgs_png_home_path = "./Data/Normal/images"
	datasetConfig.covid_imgs_png_home_path = "./Data/COVID/images"
	print(_data_config)

	# Train data loader
	datasetConfig.type = 'train'
	_train_dl = init_dataloader(_loader_config, _data_config)

	# Valid data loader
	datasetConfig.type = 'valid'
	_valid_dl = init_dataloader(_loader_config, _data_config)

	# Test data loader
	datasetConfig.type = 'test'
	_test_dl = init_dataloader(_loader_config, _data_config)

	# Config training
	modelConfig.gpu = _cuda
	modelConfig.name = backbone
	modelConfig.valid_epochs = "1_1"
	modelConfig.early_stopping = 10
	modelConfig.save_epochs = 5
	modelConfig.number_of_output_classes = 1
	modelConfig.learning_rate = _lr
	modelConfig.opt_name = 'ADAMW'
	modelConfig.epochs = 200
	modelConfig.wandb = False
	modelConfig.info = _info
	modelConfig.image_dimension = _image_size

	# Set augumentation method
	modelConfig.augumentation_model = "COVID19_GRAY"
	modelConfig.pretrained = False
	print(_model_config)

	# Run training
	_training = model_training_app(_train_dl, _valid_dl, _model_config, f'MODELS/{backbone}/{_info}/')

	if backbone == 'res18':
		_training.transfer_weights("./TL_Weights/ResNet18.pth")
	elif backbone == 'mobileNetV3Large':
		_training.transfer_weights("./TL_Weights/MobileNetV3Large.pth")
	_training.freeze_unfreeze_model(freeze = False)
	_training.start_training()


_data_partitions = [0.05, 0.25, 0.5, None]
_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
# train on this specific settings just to show minimal working sample
_learning_rates = [1e-3]
_data_partitions = [0.05]

# how many times to run the training process
_nr_runs = 1
_start_idx = 1
for run_idx in range(_start_idx,_nr_runs+_start_idx):
	for backbone in [
			'mobileNetV3Large',
			'res18',
		]:
		for _partition in _data_partitions:
			for _lrate in _learning_rates:
				process = Process(target=task, args=(_lrate, _partition, run_idx, backbone))
				process.start()
				process.join()
	
