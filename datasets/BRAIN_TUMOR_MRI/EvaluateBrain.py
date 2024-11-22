import os
from Utils.dataloader_BrainTumorMRI import *
from Utils.models import *
from Utils.TrainModel import *
from Utils.config import *
from multiprocessing import Process


# Load configs 
def task(backbone:str, model_path:str):
    _data_config = DatasetConfig()
    _loader_config = LoaderConfig()
    _model_config = ModelConfig()

    # Set root dir
    _cuda = 'cuda:0'
    _data_config.gpu = _cuda
    _data_config.image_dimension = get_image_size_for_backbone(backbone=backbone)

    _data_config.train_imgs_home_path = "./Data/Training"
    _data_config.test_imgs_home_path = "./Data/Testing"
    print(_data_config)

    _model_config.pretrained = True

    _data_config.type = 'train'
    _train_data_loader = init_dataloader(_loader_config, _data_config)
    _data_config.type = 'valid'
    _valid_data_loader = init_dataloader(_loader_config, _data_config)
    _data_config.type = 'test'
    _test_data_loader = init_dataloader(_loader_config, _data_config)

    # Config training
    _model_config.info = f'EVAL-{backbone}'
    _model_config.name = backbone
    _model_config.valid_epochs = "1_1"
    _model_config.learning_rate = 1e-4#"Auto" #"Auto" #1e-3  # not important here
    _model_config.opt_name = 'ADAMW'
    _model_config.gpu = _cuda
    _model_config.number_of_output_classes = 4
    print(_model_config)

    _model_config.pretrained = False

    # Set augumentation method
    # use RGB aug model only if imagenet!! (that's because of imagenet 3-channel scaling)
    _model_config.augumentation_model = "BRAIN_RGB" if model_path.lower().find("imagenet") != -1 else "BRAIN_GRAY"
    assert model_path.find(_model_config.name) != -1, f"Config set to model {_model_config.name}, but model path set to {model_path}!"

    print(_model_config)
    _training = model_training_app(_train_data_loader, _valid_data_loader, _model_config, os.path.join("Results_Train_Valid_Test", dirpath)+'/')
    _training.load_model(model_path)
    _training.model_predict_from_dl(_train_data_loader,"train")
    _training.model_predict_from_dl(_valid_data_loader,"valid")
    _training.model_predict_from_dl(_test_data_loader,"test")


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
    MODELS_DIR = os.path.join("MODELS", backbone)
    for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
        best_model_file = [filename for filename in filenames if filename.endswith('best_model.pth')]
        if best_model_file.__len__() != 1:
            print(f'Either there were multiple best_model files, or none were found. Skipping "{dirpath}"')
            continue
        else:
            # there should only be one file
            best_model_file = best_model_file[0]
            _FILE_PATH = os.path.join(dirpath, best_model_file)
            _TL_MODEL_NAME = [m for m in ['ImageNet', 'RadiologyNet', 'SCRATCH'] if _FILE_PATH.find(m) != -1][0]
            print(f'>> {_FILE_PATH} :: Using {_TL_MODEL_NAME} TL MODEL !!')
            process = Process(target=task, args=(backbone, _FILE_PATH))
            process.start()
            process.join()

        # also find the xlsx files which contain the training progress
        xslx_train_progress = [filename for filename in filenames if filename.endswith('.xlsx')]
        for filename in xslx_train_progress:
            _FILE_PATH = os.path.join(dirpath, filename)
            __new_pth =  os.path.join("Results_Train_Valid_Test", dirpath, f'{filename}_training.xlsx')
            print(f'Found file {filename}. Copying it to {__new_pth}')
            shutil.copy(_FILE_PATH, __new_pth)
            del __new_pth
