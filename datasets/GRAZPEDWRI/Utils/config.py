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
    labels_csv_path = None


    '''
    Imgs_png_home_path = Root directory where the images are located
    '''
    imgs_png_home_path = None

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
    Undersample: equalise all data in training set to have same representation
    by oversampling the data where necessary
    '''
    undersample = True

    '''
    Image dimension: Image size, Single integer controling the shape of the input
    image. image_dimension = 200 means that input image will be 200x200 pixels 
    '''
    image_dimension = 224

    '''
    partition: float, partition * len(dataset). Defines what partition of the given
    dataset will be used
    '''
    partition = None

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'imgs_png_home_path: {datasetConfig.imgs_png_home_path}, '+ 
            f'labels_csv_path: {labels_csv_path}, ' + 
            f'split_ratio: {datasetConfig.split_ratio}, '+
            f'type: {datasetConfig.type}, '+ 
            f'image_dimension: {datasetConfig.image_dimension}, '+
            f'number_of_labels = {datasetConfig.number_of_labels}, '+
            f'undersample = {datasetConfig.undersample}, ' +
            f'partition = {datasetConfig.partition}\n')
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
    name: set model name.
    '''
    name = 'vgg'

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
    opt_name = 'ADAMW'

    '''
    learning_rate: set a float which is the inital learning rate.
    Set it to "Auto" to find best lr automatically
    '''
    learning_rate = 10e-3

    '''
    loss_name: name of the loss.
    '''
    loss_name = 'bc_entropy'

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
    info
    '''
    info = None

    '''
    number of output classes
    '''
    number_of_output_classes = 1

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
              f'pretrained: {modelConfig.pretrained}, ' +
              f'number_of_output_classes: {modelConfig.number_of_output_classes}, ' +
              f'info: {modelConfig.info}, ' +
              f'wandb: {modelConfig.wandb}\n')
        _retval += ("######################################\n")
        return _retval
        
        
def get_image_size_for_backbone(backbone: str):
    """
    Get the expected image size for different topologies.
    The images are expected to be squares, so this will return only
    a single number, indicating both height and width.
    """
    img_size = 224    
    return img_size

