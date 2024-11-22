#*******************************************************#
# Classes used as configuration files
#*******************************************************#  

class DatasetConfig:
    '''
    Class for defining dataset parameters: shape, split ratio, augumentation etc.
    '''
    '''
    Imgs_png_home_path = Root directory where the images are located
    '''
    train_imgs_home_path = "../Data/Training"
    test_imgs_home_path = "../Data/Testing"

    split_ratio = 0.75
    '''
    Split ratio: A single floating point number representing
    the ratio between train, validation and test set. For instance:
    split_ratio = 0.8, then 80% of data is assigned to training, the remainder
    is split between validation (10% of total data) and test (10% of total data).
    '''

    type = 'Train'
    '''
    Type: Type of the dataset which is being returned: 'train', 'valid' or 'test'
    This is important for data sampling to be always the same
    '''

    image_dimension = 224
    '''
    Image dimension: Image size, Single integer controling the shape of the input
    image. image_dimension = 200 means that input image will be 200x200 pixels 
    '''

    partition = None
    '''
    partition: float, partition * len(dataset). Used to reduce the training data size.
    For example, if partition = 0.05, then 5% of the total available training data will be used.
    If None, the training data pool will not be reduced.
    '''

    number_of_labels = 1
    '''
    This indicate how many classes there are to predict.
    '''

    def __str__(self):
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'train_imgs_png_home_path: {self.train_imgs_home_path}, '+ 
            f'test_imgs_png_home_path: {self.test_imgs_home_path}, '+
            f'split_ratio: {self.split_ratio}, '+
            f'type: {self.type}, '+ 
            f'image_dimension: {self.image_dimension}, '+
            f'number_of_labels = {self.number_of_labels}, '+
            f'partition = {self.partition}\n')
        _retval += ("######################################\n")
        return _retval
    
class LoaderConfig:
    '''
    Class for defining loader parameters: batch_size, number of workers
    and gpu
    '''

    batch_size = 32
    '''
    batch_size: define size of the batch
    '''

    number_of_workers = 4
    '''
    number_of_workers: paralelization for data loading
    '''

    use_gpu = 'cpu'
    '''
    use_gpu: name of the gpu: typicaly 'cuda:0', 'cuda:1' or 'cpu'
    '''

    def __str__(self):
        _retval = ''
        _retval += "######################################\n"
        _retval += (f'batch_size: {self.batch_size}, '+
              f'number_of_workers: {self.number_of_workers}, ' +
              f'use_gpu: {self.use_gpu}\n')
        _retval += "######################################\n"
        return _retval
    

class ModelConfig:
    '''
    Class for defining model parameters: model name, number of epoch,
    validation ratio, early stopping, optimizer, learning rate, 
    loss, and gpu
    '''

    name = 'res50'
    '''
    name: set model name. by setting this parameter, we can pick the right topology for training.
    '''

    epochs = 200
    '''
    epochs: number of epochs for training
    '''

    valid_epochs = '2_10'
    '''
    valid_epochs: number which defines how often validation will occur. Format is
    "x_y" where model performers validation every x epochs until it reaches the y-th epoch. 
    After y-th epoch is reached, validation is performed every epoch.
    '''

    save_epochs = 5
    '''
    save_epochs: number after each training model is being saved
    '''

    early_stopping = 10
    '''
    early_stopping: define early stopping besed on the validation epochs
    '''

    opt_name = 'ADAMW'
    '''
    opt_name: optimizer name. Currently implemented: "ADAMW" and "ADAM"
    '''

    learning_rate = 10e-3
    '''
    learning_rate: set a float which is the inital learning rate.
    '''

    scheduler = 'ReduceLROnPlateau'
    '''
    Scheduler: implemented schedulers ReduceLROnPlateau, #CyclicLRWithRestarts
    '''

    loss_name = 'c_entropy'
    '''
    loss_name: name of the loss.
    '''

    gpu = 'cpu'
    '''
    gpu: define which gpu will be use. Chose from names: 'cuda:0', 'cuda:1', 'cpu'
    '''

    augumentation_model = None
    '''
    Augumentation model: model given for the augumentation. Several
    example models are located in models.py. E.G.: RGB_transform, TransformXrayGray
    '''

    pretrained = False
    '''
    Pretrained: If model wants to be trained from scratch, set this to False, otherwise
    ImageNet or RadiologyNET weights will be loaded.
    '''

    info = None
    '''
    str param used just for logging
    '''

    number_of_output_classes = 4
    '''
    How many labels to predict.
    '''
    
    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += (f'name: {self.name}, '+ 
              f'epochs: {self.epochs}, ' +
              f'valid_epochs: {self.valid_epochs}, '+
              f'early_stopping: {self.early_stopping}, '+ 
              f'opt_name: {self.opt_name}, '+ 
              f'save_epochs: {self.save_epochs}, '+ 
              f'learning_rate: {self.learning_rate}, '+ 
              f'loss_name: {self.loss_name}, '+ 
              f'augumentation_model: {self.augumentation_model}, ' +
              f'gpu: {self.gpu}, ' +
              f'pretrained: {self.pretrained}, ' +
              f'number_of_output_classes: {self.number_of_output_classes}, ' +
              f'info: {self.info}\n')
        _retval += ("######################################\n")
        return _retval


def get_image_size_for_backbone(backbone: str):
    """
    Get the expected image size for different topologies.
    The images are expected to be squares, so this will return only
    a single number, indicating both height and width.

    When it comes to Brain Tumor, all tested architectures are fed 224x224 images.
    """
    if backbone in ['mobileNetV3Small', 'mobileNetV3Large', 'res50', 'res18', 'res34', 'dense121', 'vgg16']:
        img_size = 224
    elif backbone in ['inceptionV3']:
        img_size = 299
    elif backbone in ['eff3']:
        img_size = 300
    elif backbone in ['eff4']:
        img_size = 380
    else:
        raise NotImplementedError(f'Unknown image input size for {backbone}')
    
    return img_size
