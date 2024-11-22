from Utils.dataloader_BrainTumorMRI import *
from Utils.config import *
from Utils.models import *
from Utils.TrainModel import *
from multiprocessing import Process

from codecarbon import EmissionsTracker
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def task(lr, data_partition:float=None, run_idx:int=None, backbone: str='dense121'):
	_cuda = 'cuda:1'
	_image_size = get_image_size_for_backbone(backbone=backbone)
	_lr = lr
	_info = "RadiologyNet_"+str(lr)+f"_{backbone}"
	_info += f'_partition-{data_partition}' if data_partition is not None else ''
	_info += f'_run-{run_idx:03}' if run_idx is not None else ''
	logger.info(_info)
	# Load configs 
	_data_config = DatasetConfig()
	_loader_config = LoaderConfig()
	_model_config = ModelConfig()

	#Create data
	_loader_config.use_gpu = _cuda
	_loader_config.number_of_workers = 4
	_loader_config.batch_size  = 32
	logger.info(_loader_config)

	# Datasets and datalaoders
	# General config
	_data_config.image_dimension = _image_size
	_data_config.train_imgs_home_path = "./Data/Training"
	_data_config.test_imgs_home_path = "./Data/Testing"
	logger.info(_data_config)

	# Train data loader
	_data_config.type = 'train'
	_data_config.partition = _partition
	_train_dl = init_dataloader(_loader_config, _data_config)

	# Valid data loader
	_data_config.type = 'valid'
	_valid_dl = init_dataloader(_loader_config, _data_config)

	# Test data loader
	_data_config.type = 'test'
	_test_dl = init_dataloader(_loader_config, _data_config)

	# Config training
	_model_config.name = f'{backbone}'
	_model_config.valid_epochs = "1_1"
	_model_config.early_stopping = 10
	_model_config.learning_rate = _lr
	_model_config.opt_name = 'ADAMW'
	_model_config.epochs = 10
	_model_config.save_epochs = 5
	_model_config.gpu = _cuda
	_model_config.number_of_output_classes = 4
	_model_config.info = _info

	# Set augumentation method
	_model_config.augumentation_model = "BRAIN_GRAY"
	_model_config.pretrained = False

	# Run training
	_training = model_training_app(_train_dl, _valid_dl, _model_config, f'MODELS/{backbone}/{_info}/')

	if backbone == 'dense121':
		_training.transfer_weights("./TL_Weights/DenseNet121.pth")
	elif backbone == 'eff3':
		_training.transfer_weights("./TL_Weights/EfficientNetB3.pth")
	elif backbone == 'eff4':
		_training.transfer_weights("./TL_Weights/EfficientNetB4.pth")
	elif backbone == 'inceptionV3':
		_training.transfer_weights("./TL_Weights/InceptionV3.pth")
	elif backbone == 'res18':
		_training.transfer_weights("./TL_Weights/ResNet18.pth")
	elif backbone == 'res34':
		_training.transfer_weights("./TL_Weights/ResNet34.pth")
	elif backbone == 'res50':
		_training.transfer_weights("./TL_Weights/ResNet50.pth")
	elif backbone == 'mobileNetV3Small':
		_training.transfer_weights("./TL_Weights/MobileNetV3Small.pth")
	elif backbone == 'mobileNetV3Large':
		_training.transfer_weights("./TL_Weights/MobileNetV3Large.pth")
	elif backbone == 'vgg16':
		_training.transfer_weights("./TL_Weights/VGG16.pth")
		
	_training.freeze_unfreeze_model(freeze = False)

	gpu_ids = [int(_cuda[-1])]
	tracker = EmissionsTracker(project_name=f'BRAIN_TUMOR__{backbone}/{_info}/', gpu_ids=gpu_ids)
	tracker.start()

	_training.start_training()

	emissions = tracker.stop()
	logger.info(emissions)


_data_partitions = [0.05, 0.25, 0.5, None]
# _learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
# train on this specific settings just to show minimal working sample
# _learning_rates = [1e-3]
# _data_partitions = [0.05]

# how many times to run the training process
_nr_runs = 1
_start_idx = 1
for run_idx in range(_start_idx,_nr_runs+_start_idx):
	for backbone in [
			'dense121',
			'eff3',
			'eff4',
			'inceptionV3',
			'mobileNetV3Small',
			'mobileNetV3Large',
			'res18',
			'res34',
			'res50',
			'vgg16',
		]:
		for _partition in _data_partitions:
			for _lrate in _learning_rates:
				process = Process(target=task, args=(_lrate, _partition, run_idx, backbone))
				process.start()
				process.join()
	
