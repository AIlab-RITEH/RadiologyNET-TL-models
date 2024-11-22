#*******************************************************#
# Classes used as configuration files
#*******************************************************#  

class datasetConfig:
    '''
    Class for defining dataset parameters: shape, split ratio, augumentation etc.
    '''
    '''
    Labes_csv_path: Root directory where the labels are located
    '''
    labels_csv_path = '/home/Shared/RadiologyNET/RadiologyNET_LABELS.csv'

    '''
    Imgs_png_home_path: Root directory where the images are located
    '''
    imgs_png_home_path = '/home/Shared/RadiologyNET/current_research/converted_images224x224'


    '''
    normalization: path to json file generated where scaling factors will be stored. 
    Defualt is None --> no saving/applaying scaling
    '''
    normalization = None
    
    '''
    partition: float, partition * len(dataset). Defines what partition of the given
    dataset will be used
    '''
    partition = None
  
    '''
    Image dimension: Image size, Single integer controling the shape of the input
    image. image_dimension = 200 means that input image will be 200x200 pixels 
    '''
    image_dimension = 500

    '''
    Verbose: printing some notifications during debug
    '''
    verbose = False
        
    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'labels_csv_path: {datasetConfig.labels_csv_path}, '+ 
            f'imgs_png_home_path: {datasetConfig.imgs_png_home_path}, '+
            f'image_dimension: {datasetConfig.image_dimension}, '+
            f'normalization: {datasetConfig.normalization}, '+
            f'verbose: {datasetConfig.verbose}\n')
        
        _retval += ("######################################\n")
        return _retval
    
class loaderConfig:
    '''
    Class for defining loader parameters: batch_size, number of workers
    and gpu
    '''
    '''
    batch_size: define size of the batch
    '''
    batch_size = 32

    '''
    number_of_workers: paralelization for data loading
    '''
    number_of_workers = 1

    '''
    use_gpu: name of the gpu: typicaly 'cuda:0', 'cuda:1' or 'cpu'
    '''
    use_gpu = 'cuda:0'


    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += "######################################\n"
        _retval += (f'batch_size: {loaderConfig.batch_size}, '+
              f'number_of_workers: {loaderConfig.number_of_workers}, ' +
              f'use_gpu: {loaderConfig.use_gpu}\n')
        _retval += "######################################\n"
        return _retval
    
class modelConfig:
    '''
    Class for defining model parameters: model name, number of epoch,
    validation ratio, early stopping, optimizer, learning rate, 
    loss, and gpu
    '''
    '''
    name: set model name. Currently implemented: 'eff', 'vgg', 'res'
    '''
    name = 'RSNA'

    '''
    backbone: select backbone for the RSNA neural network. Default is paper
    default: inceptionV3. Available: inceptionV3, resNet50
    '''
    backbone = 'inceptionV3'

    '''
    epochs: number of epochs for training
    '''
    epochs = 150

    '''
    valid_epochs: number which defines when validation will occure. Format is
    "x_y" where model does each "x" epochs validaiton until it reaches "y" epoch. 
    After "y" epoch is reached, validation is done every epoch
    '''
    valid_epochs = '2_10'

    '''
    valid_percent: percentage of trainset passed after which validation will be performed.
    This is usefull for big datasets.
    '''
    valid_percent = None

    '''
    image_dimension: model's input size. 200 means image 200x200
    '''
    image_dimension = 500
  
    '''
    save_epochs: number after each training model is being saved
    '''
    save_epochs = 5

    '''
    early_stopping: define early stopping besed on the validation epochs
    '''
    early_stopping = 50

    '''
    opt_name: optimizer name. Currently implemented: "ADAMW" and "ADAM"
    '''
    opt_name = 'ADAM'

    '''
    learning_rate: set a float which is the inital learning rate.
    Set it to "Auto" to find best lr automatically
    '''
    learning_rate = 10e-3

    '''
    Scheduler: implemented schedulers ReduceLROnPlateau, #CyclicLRWithRestarts
    '''
    scheduler = 'ReduceLROnPlateau'

    '''
    loss_name: name of the loss. Currently implemented: 'c_entropy', 'mse'
    '''
    loss_name = 'mse'

    '''
    gpu: define which gpu will be use. Chose from names: 'cuda:0', 'cuda:1', 'cpu'
    '''
    gpu = 'cpu'

    '''
    Augumentation model: model given for the augumentation. Several
    example models are located in models.py. E.G.: RGB_transform, TransformXrayGray
    '''
    augumentation_model = None

    '''
    Wandb: True or False if someone wants wandb monitoring
    '''
    wandb = False

    '''
    Pretrained: If model wants to be trained from scratch, set this to False, otherwise
    ImageNet weights will be loaded
    '''
    pretrained = False

    '''
    Custom_info: custom information to be stored/displayed in wandb
    '''
    custom_info = None


    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'name: {modelConfig.name}, '+ 
              f'epochs: {modelConfig.epochs}, ' +
              f'valid_epochs: {modelConfig.valid_epochs}, '+
              f'early_stopping: {modelConfig.early_stopping}, '+ 
              f'opt_name: {modelConfig.opt_name}, '+ 
              f'save_epochs: {modelConfig.save_epochs}, '+ 
              f'learning_rate: {modelConfig.learning_rate}, '+ 
              f'loss_name: {modelConfig.loss_name}, '+ 
              f'augumentation_model: {modelConfig.augumentation_model}, ' +
              f'gpu: {modelConfig.gpu}, ' +
              f'valid_percent: {modelConfig.valid_percent}, ' +
              f'pretrained: {modelConfig.pretrained}, ' +
              f'custom_info: {modelConfig.custom_info}, '+
              f'wandb: {modelConfig.wandb}\n')
        _retval += ("######################################\n")
        return _retval
