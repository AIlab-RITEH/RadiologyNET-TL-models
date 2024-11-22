from Utils.dataloader_GRAZPEDWRI import *
from Utils.config import *
from Utils.models import *
from Utils.TrainModel import *
from multiprocessing import Process

def task(lr, data_partition:float=None, run_idx:int=None, backbone: str='eff0'):
	_cuda = 'cuda:0'
	_image_size = get_image_size_for_backbone(backbone=backbone)
	_lr = lr
	_info = "SCRATCH_"+str(lr)+f"_{backbone}"
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
	datasetConfig.partition = _partition
	print(_loader_config)

	# Training dataset
	datasetConfig.imgs_png_home_path = './Data/Images'
	datasetConfig.labels_csv_path = './Data/dataset.csv'
	datasetConfig.undersample = True
	
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
	modelConfig.augumentation_model = "GRAZPEDWRI_GRAY"
	modelConfig.pretrained = False
	print(_model_config)

	_training = model_training_app(_train_dl, _valid_dl, _model_config, f'MODELS/{backbone}/{_info}/')
	_training.freeze_unfreeze_model(freeze = False)
	_training.start_training()



_data_partitions = [0.05, 0.25, 0.5, None]
_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]

# adjust these just for testing and brevity
_data_partitions = [0.05]
_learning_rates = [1e-3]

# how many times to run the training process
_nr_runs = 1
_start_idx = 1
for run_idx in range(_start_idx,_nr_runs+_start_idx):
	for backbone in [
			'dense121',
			'res34',
	]:
		for _partition in _data_partitions:
			for _lrate in _learning_rates:
				# task(_lrate, _partition, run_idx, backbone)
				process = Process(target=task, args=(_lrate, _partition, run_idx, backbone))
				process.start()
				process.join()
		
