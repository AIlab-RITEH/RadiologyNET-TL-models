import os
import gensim
from Utils.dataloader_RSNA_BONE import *
from Utils.models import *
from Utils.TrainModelRSNA import *
from Utils.config import *
from multiprocessing import Process

# OPTIONAL: Load the "autoreload" extension so that code can change
#%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
#%autoreload 2

def task(backbone: str, model_path:str):
    # Load configs 
    _data_config = datasetConfig()
    _loader_config = loaderConfig()
    _model_config = modelConfig()

    #Create data
    loaderConfig.use_gpu = 'cuda:0'
    loaderConfig.number_of_workers = 4
    loaderConfig.batch_size  = 32
    print(_loader_config)
       
    if backbone == 'eff3':
        datasetConfig.image_dimension = 300
    elif backbone == 'resNet50':
        datasetConfig.image_dimension = 224
    elif backbone == 'inceptionV3':
        datasetConfig.image_dimension = 299
    else:
        raise NotImplementedError(f'Unknown backbone {backbone}')

    modelConfig.backbone = backbone
    modelConfig.image_dimension = datasetConfig.image_dimension

    # Training dataset
    datasetConfig.labels_csv_path = "Data/TrainingSet/train.csv"
    datasetConfig.imgs_png_home_path = "Data/TrainingSet/Images"
    # datasetConfig.partition = 0.2
    print(_data_config)
    # Train data loader
    _train_dl = init_dataloader(_loader_config, _data_config)

    # Validation dataset
    datasetConfig.labels_csv_path = "Data/ValidationSet/Valid.csv"
    datasetConfig.imgs_png_home_path = "Data/ValidationSet/Images"
    # Valid data loader
    _valid_dl = init_dataloader(_loader_config, _data_config)

    # Test dataset
    datasetConfig.labels_csv_path = "Data/TestSet/Test.csv"
    datasetConfig.imgs_png_home_path = "Data/TestSet/Images"
    print(_data_config)
    # test data loader
    _test_dl = init_dataloader(_loader_config, _data_config)

    # Config training
    modelConfig.gpu = loaderConfig.use_gpu
    modelConfig.name = 'RSNA'
    modelConfig.loss = 'mse'
    modelConfig.valid_epochs = "1_1"
    modelConfig.early_stopping = 100
    modelConfig.learning_rate = 1e-5#"Auto" #"Auto" #1e-3
    modelConfig.opt_name = 'ADAM'
    modelConfig.epochs = 500
    modelConfig.wandb = False
    modelConfig.custom_info = 'RadiologyNetEval'


    # Set augumentation method
    # modelConfig.augumentation_model = "RSNA-GRAY" #"XRAY_transform_GRAY" #"RSNA"#"XRAY_transform_RGB"#"GRAY_transform"#"XRAY_transform_GRAY" #None #'XRAY_transform_RGB'
    modelConfig.augumentation_model = "RSNA-RGB" if 'imagenet' in model_path.lower() else 'RSNA-GRAY'
    print('#### ', model_path,  '<<<<<< model >>>>>>', _model_config.augumentation_model, '<<<<< aug model')
    _training = model_training_app(_train_dl, _valid_dl, _model_config, os.path.join("Results_Train_Valid_Test", dirpath)+'/')
    _training.load_model(model_path)
    _training.model_predict_from_dl(_train_dl,"train")
    _training.model_predict_from_dl(_valid_dl,"valid")
    _training.model_predict_from_dl(_test_dl,"test")


for backbone in [
        'eff3',
        'inceptionV3'
    ]:
    MODELS_DIR = os.path.join("MODELS", backbone)
    for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
        if backbone == 'inceptionV3' and 'imagenet' in dirpath.lower():
            # inference for imagenet inceptv3 models is done immediately after training
            print(f'Skipping {dirpath}')
            continue

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

            task(backbone=backbone, model_path=_FILE_PATH)

        # also find the xlsx files which contain the training progress
        xslx_train_progress = [filename for filename in filenames if filename.endswith('.xlsx')]
        for filename in xslx_train_progress:
            _FILE_PATH = os.path.join(dirpath, filename)
            __new_pth =  os.path.join("Results_Train_Valid_Test", dirpath, f'{filename}_training.xlsx')
            print(f'Found file {filename}. Copying it to {__new_pth}')
            shutil.copy(_FILE_PATH, __new_pth)
            del __new_pth


