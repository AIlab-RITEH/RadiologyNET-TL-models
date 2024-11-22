def get_image_size_for_backbone(backbone: str):
    """
    Get the expected image size for different topologies.
    The images are expected to be squares, so this will return only
    a single number, indicating both height and width.
    """
    img_size = 224
    if backbone in ['eff4']:
        img_size = 380
    return img_size



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
    Imgs_png_home_path = Root directory where the images are located
    '''
    imgs_png_home_path = '/home/Shared/RadiologyNET/current_research/converted_images224x224'
    
    '''
    Split ratio: A single floating point number representing
    the ration between train, validation and test set. For instance:
    split_ratio = 0.8, 0.8 is to train set, 0.1 to valid set 
    and 0.1 to test set.
    '''
    split_ratio = 0.75

    '''
    Type: Type of the dataset which is being returned: 'train', 'valid' or 'test'
    This is important for data sampling to be always the same
    '''
    type = 'Train'

    '''
    Threshold: Number which controlls frequency of each label in the dataset.
    For instance: Threshold = 20 means that each label must be at least 20 represented
    in the dataset
    '''
    threshold = 20

    '''
    Image dimension: Image size, Single integer controling the shape of the input
    image. image_dimension = 200 means that input image will be 200x200 pixels 
    '''
    image_dimension = 512

    '''
    Verbose: printing some notifications during debug
    '''
    verbose = False

    '''
    Oversample: equalise all data in training set to have same representation
    by oversampling the data where necessary
    '''
    oversample = False

    '''
    Blacklist: Provide a list of labels which will be ignored. Otherwise set to empty list
    '''
    blacklist = []

    '''
    Number of labels: Provide a number of labeles-necessary for the black list
    '''
    number_of_labels = 50

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'labels_csv_path: {datasetConfig.labels_csv_path}, '+ 
            f'imgs_png_home_path: {datasetConfig.imgs_png_home_path}, '+
            f'split_ratio: {datasetConfig.split_ratio}, '+
            f'type: {datasetConfig.type}, '+ 
            f'threshold: {datasetConfig.threshold}, ' +
            f'image_dimension: {datasetConfig.image_dimension}, '+
            f'oversample: {datasetConfig.oversample}, ' + 
            f'verbose: {datasetConfig.verbose}\n'
            )
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
    number_of_workers = 4

    '''
    use_gpu: name of the gpu: typicaly 'cuda:0', 'cuda:1' or 'cpu'
    '''
    use_gpu = 'cpu'


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
    name: set model name. Currently implemented
    '''
    name = 'UNet'

    '''
    backbone: set backbone for the nn. Currently implemented: res50, eff4, vgg16
    '''
    backbone = 'res50'

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
    save_epochs: number after each training model is being saved
    '''
    save_epochs = 1

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
    loss_name: name of the loss. Currently implemented: 'c_entropy', 'mse'
    '''
    loss_name = 'dice_loss'

    '''
    gpu: define which gpu will be use. Chose from names: 'cuda:0', 'cuda:1', 'cpu'
    '''
    gpu = 'cpu'

    '''
    Augumentation model: model given for the augumentation. Several
    example models are located in models.py. E.G.: RGB_transform, TransformXrayGray
    '''
    augumentation_model = 'LUNA'

    '''
    number_of_labels: number of output channels, default 1
    '''
    number_of_labels = 1
    
    '''
    Everything for augumentation based on repo form the book
    '''
    flip = True
    offset = 0.03
    scale = 0.2
    rotate = True
    noise = 25.0

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
    weights_path: path to weights for transfer learning
    '''
    weights_path = None

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
              f'backbone: {modelConfig.backbone, }'+
              f'weights_path:{modelConfig.weights_path, }' +
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
              f'custom info: {modelConfig.custom_info}, ' +
              f'number_of_labels: {modelConfig.number_of_labels}, ' +
              f'wandb: {modelConfig.wandb}\n')
        _retval += ("######################################\n")
        return _retval