from luna.paths import LunaPaths
from luna import candidate_info, dataset
from luna.ct import Ct
from luna.prepcache import LunaPrepCacheApp
from torch.utils.data import DataLoader
#from Utils.dataloader import *
from Utils.models import *
from Utils.TrainModelLUNA import *
from Utils.config import *
from multiprocessing import Process

def task(lr, run_idx:int=None, backbone: str='res50', freeze:bool=False):
	_cuda = 'cuda:2'
	_image_size = get_image_size_for_backbone(backbone=backbone)
	_lr = lr
	_info = "SCRATCH_"+str(lr)+f"_{backbone}"
	_info += f'_frozen' if freeze == True else ''
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
	
	
	# Dataset
	# Set up dir paths
	LUNA_ROOT = ('LUNADATASET/unzipped')
	LUNA_PATHS = LunaPaths(luna_root=LUNA_ROOT)
	
	# Data precache
	NUM_WORKERS = 8
	BATCH_SIZE = 32
	#how many context slices to use when cropping areas of interest
	CONTEXT_SLICES_COUNT = 0
	
	# Training dataset
	luna_train_ds = dataset.TrainingLuna2dSegmentationDataset(
    		is_val_set=False,
    		contextSlices_count=CONTEXT_SLICES_COUNT,
    		val_stride=10
	)
	
	# Validation dataset
	luna_val_ds = dataset.Luna2dSegmentationDataset(
    		is_val_set=True,
    		contextSlices_count=CONTEXT_SLICES_COUNT,
    		val_stride=10
	)
	# Data loaders
	_train_dl = DataLoader(luna_train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
	_valid_dl = DataLoader(luna_val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
	
	# Config training
	modelConfig.gpu = _cuda
	modelConfig.name = 'UNET'
	modelConfig.backbone = backbone
	modelConfig.weights_path = None
	modelConfig.loss = 'dice'
	modelConfig.valid_epochs = "1_1"
	modelConfig.early_stopping = 10
	modelConfig.save_epochs = 1
	modelConfig.learning_rate = _lr
	modelConfig.opt_name = 'ADAM'
	modelConfig.epochs = 200
	modelConfig.custom_info = _info
	modelConfig.image_dimension = _image_size

	# Set augumentation method
	modelConfig.augumentation_model = "LUNA_GRAY"

	# Pretrained
	modelConfig.pretrained = False
	print(_model_config)

	_training = model_training_app(_train_dl, _valid_dl, _model_config, f'{backbone}/{_info}/')
	_training.freeze_unfreeze_model(freeze = False)
	_training.start_training()


_learning_rates = [1e-3, 1e-4, 1e-5]
_nr_runs = 1
_run_start_idx = 1
for backbone in [
	'vgg16',
	'eff4',
	'res50',
]:
	for freeze in [False]:
		for _lrate in _learning_rates:
			for run_idx in range(_run_start_idx,_nr_runs+_run_start_idx):
				process = Process(target=task, args=(_lrate, run_idx, backbone, freeze))
				process.start()
				process.join()
	
