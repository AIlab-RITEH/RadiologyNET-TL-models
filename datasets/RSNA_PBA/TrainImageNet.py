from Utils.dataloader_RSNA_BONE import *
from Utils. config import *
from Utils.models import *
from Utils.TrainModelRSNA import *
from multiprocessing import Process
from codecarbon import EmissionsTracker

def task(lr, data_partition:float=None, run_idx:int=None, backbone: str='eff3'):
	_cuda = 'cuda:0'
	if backbone == 'eff3':
		_image_size = 300
	elif backbone == 'resNet50':
		_image_size = 224
	elif backbone == 'inceptionV3':
		_image_size = 299

	_lr = lr
	_info = "ImageNet_"+str(lr)+f"_{backbone}"
	_info += f'_partition-{round(data_partition,2)}' if data_partition is not None else ''
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
	datasetConfig.labels_csv_path = "./Data/TrainingSet/train.csv"
	datasetConfig.imgs_png_home_path = "./Data/TrainingSet/Images"
	datasetConfig.image_dimension = _image_size
	print(_data_config)
	# Train data loader
	_train_dl = init_dataloader(_loader_config, _data_config)

	# Validation dataset
	datasetConfig.labels_csv_path = "./Data/ValidationSet/Valid.csv"
	datasetConfig.imgs_png_home_path = "./Data/ValidationSet/Images"
	datasetConfig.image_dimension = _image_size
	# Valid data loader
	_valid_dl = init_dataloader(_loader_config, _data_config)

	# Test dataset
	datasetConfig.labels_csv_path = "Data/TestSet/Test.csv"
	datasetConfig.imgs_png_home_path = "Data/TestSet/Images"
	print(_data_config)
	# test data loader
	_test_dl = init_dataloader(_loader_config, _data_config)

	# Config training
	modelConfig.gpu = _cuda
	modelConfig.name = 'RSNA'
	modelConfig.loss = 'mse'
	modelConfig.valid_epochs = "1_1"
	modelConfig.early_stopping = 10
	modelConfig.save_epochs = 1
	modelConfig.learning_rate = _lr
	modelConfig.opt_name = 'ADAM'
	modelConfig.epochs = 200
	modelConfig.wandb = False
	modelConfig.custom_info = _info
	modelConfig.backbone = backbone
	modelConfig.image_dimension = _image_size

	# Set augumentation method
	modelConfig.augumentation_model = "RSNA-RGB"

	# Pretrained
	modelConfig.pretrained = True
	print(_model_config)

	if os.path.exists(f'{backbone}/{_info}/'):
		print(f'Path {backbone}/{_info}/ already exists. Skipping...')
		return

	_training = model_training_app(_train_dl, _valid_dl, _model_config, f'MODELS/{backbone}/{_info}/')
	_training.freeze_unfreeze_model(freeze = False)

	#_training.transfer_weights("/home/franko/Desktop/CurrentResearch/RadiologyNet/RSNA-CLEARED/incept_v3_ADAMW_0.001_c_entropyinterim_best_model.pth")
	#_training.freeze_unfreeze_model(freeze = False)

	gpu_ids = [int(_cuda[-1])]
	tracker = EmissionsTracker(project_name=f'RSNA_BONE__{backbone}/{_info}/', gpu_ids=gpu_ids)
	tracker.start()

	_training.start_training()

	emissions = tracker.stop()
	print(emissions)

	if backbone == 'inceptionV3':
		# once training is finished, run inference
		_training.epoch = 10  # not important, it's here just so the code doesn't break
		_training.model_predict_from_dl(_train_dl,"train")
		_training.model_predict_from_dl(_valid_dl,"valid")
		_training.model_predict_from_dl(_test_dl,"test")

		# once inference is over, copy the predictions to results output dir Results_Train_Valid_Test/
		INCEPTV3_OUTPUT_DIR =  os.path.join(f'{modelConfig.backbone}/{_info}/')
		os.makedirs(os.path.join("Results_Train_Valid_Test", INCEPTV3_OUTPUT_DIR), exist_ok=True)
		xslx_train_progress = [filename for filename in os.listdir(_training.results_output_dir) if filename.endswith('.xlsx')]
		for filename in xslx_train_progress:		
			_FILE_PATH = os.path.join(_training.results_output_dir, filename)
			if filename.endswith('train.xlsx') or filename.endswith('valid.xlsx') or filename.endswith('test.xlsx'):
				__new_pth =  os.path.join("Results_Train_Valid_Test", INCEPTV3_OUTPUT_DIR, filename)
			else:
				__new_pth =  os.path.join("Results_Train_Valid_Test", INCEPTV3_OUTPUT_DIR, f'{filename}_training.xlsx')
			print(f'Found file {filename}. Copying it to {__new_pth}')
			shutil.copy(_FILE_PATH, __new_pth)
			del __new_pth


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
		'eff3',
		'inceptionV3',
	]:
		for _partition in _data_partitions:
			for _lrate in _learning_rates:
				# task(_lrate, _partition, run_idx, backbone)
				process = Process(target=task, args=(_lrate, _partition, run_idx, backbone))
				process.start()
				process.join()