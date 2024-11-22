from collections import namedtuple
import torch
from Utils_ImageNet.models import *
import numpy as np
import datetime
import pandas as pd
import xlsxwriter
from sklearn import metrics
import time
import shutil
import os
import wandb
from Utils_ImageNet.dataloader import *
from Utils_ImageNet.config import loaderConfig
from torch_lr_finder import LRFinder

class model_training_app:
    
    def __init__(self, train_dl, valid_dl, model_params, results_output_dir):
        """
        init training with given params
        
        Args:
            * train_dl, train data set dataloader
            * valid_dl, validation data set dataloader
            * model_params, name+model params. Names: "vgg, eff, res"
            * results_output_dir, str, path to output dir for results
        """
        print("**************************************")

        # Define output dir (delete if exists + create new one)
        self.results_output_dir = results_output_dir
        if os.path.exists(self.results_output_dir):
            shutil.rmtree(self.results_output_dir)
        os.makedirs(self.results_output_dir)

        # Load data
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model_params = model_params

        # Set device
        if self.model_params.gpu == False or self.model_params.gpu == 'cpu' :
            self.device = torch.device("cpu")
        else:
            self.use_cuda = torch.cuda.is_available()        
            self.device = torch.device(self.model_params.gpu)
        
        # Get model, Optimizer and augumentation model
        self.model = self.init_model()
        

        # Augumentation model
        self.aug_model = None
        if self.model_params.augumentation_model != None:
            self.init_augumentation_model()           

        # Loss function
        self.loss = self.init_loss()

        # Init optimzer (must be last for lr find)
        self.optimizer = self.init_optimizer()

        

        # Wandb
        if self.model_params.wandb == True:
            self.init_wandb()

        print("**************************************")

    #*******************************************************#
    # Inti augumentation model
    #*******************************************************# 
    def init_augumentation_model(self):
        """
        Script which inits augumentation model
        """
        self.aug_model_color_only = TransformToRGB2()

        if self.model_params.augumentation_model == 'RGB_transform':
            self.aug_model = TransformToRGB()          
            
        if self.model_params.augumentation_model == 'XRAY_transform_RGB':
            self.aug_model = TransformXrayRGB2()

        if self.model_params.augumentation_model == 'XRAY_transform_GRAY':
            self.aug_model = TransformXrayGray()
            self.aug_model_color_only = AddChanels()

        if self.model_params.augumentation_model == 'GRAY_transform':
            self.aug_model = AddChanels()
            self.aug_model_color_only = AddChanels()
        
        


    #*******************************************************#
    # Wandb init script
    #*******************************************************# 
    def init_wandb(self, time_stamp = None, name = "Stage-0"):
        """
        Init function for wandb

        Args:
            * name, name of the training
        """
        # Obtain time
        if time_stamp == None:
            _current_time = time.strftime("%H_%M_%S", time.localtime())
        else:
            _current_time = time_stamp

        # Obtain augumentation
        if self.model_params.augumentation_model != None:
            _aug = True
        else:
            _aug = False


        wandb.init(
        # set the wandb project where this run will be logged
        project=f"ImageNet_{self.model_params.name}_"+f"{_current_time}",
        name = f"{name}",
        # track hyperparameters and run metadata
        config={f'name': self.model_params.name,
              f'custom_info': self.model_params.custom_info,   
              f'epochs': self.model_params.epochs,
              f'valid_epochs': self.model_params.valid_epochs, 
              f'early_stopping': self.model_params.early_stopping, 
              f'opt_name': self.model_params.opt_name,
              f'learning_rate': self.model_params.learning_rate, 
              f'loss_name': self.model_params.loss_name, 
              f'augumentation_model': _aug,
              f'gpu': self.model_params.gpu
            }
        )
     


    #*******************************************************#
    # Model handling scripts
    #*******************************************************#   
    def init_model(self):
        """
        Model is initially unfrozen
        """
        assert self.model_params.name in ["UNet","eff", "vgg", "res", "incept_v3"], f"Wrong model name, got: {self.model_params.name}"
        print(f"USING MODEL: {self.model_params.name}, WEIGHTS: UNFROZEN")

        if self.model_params.name == 'vgg':
            _model = VGG16(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        if self.model_params.name == 'eff':
            _model = EfficientNetB3(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        if self.model_params.name == 'res':
            _model = ResNet50(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)


        if self.model_params.name == 'incept_v3':
            _model = InceptionV3(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)

        if self.model_params.name == 'UNet':
            _model = UNet(in_channels = self.model_params.in_channels,
                          n_classes = self.model_params.n_classes,
                          depth = self.model_params.depth,
                          wf = self.model_params.wf,
                          padding = self.model_params.padding,
                          batch_norm = self.model_params.batch_norm,
                          up_mode = self.model_params.up_mode)
        # Send it to gpu
        if self.model_params.gpu != False and self.model_params.gpu != "cpu":
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            print("USING CPU")

        return _model
    
    def load_model(self, path):
        """
            Function that loads model.

            Args:
                * path, string, path to the model checkpoint
        """
        print("LOADING MODEL")
        
        _state_dict = torch.load(path)
        self.model.load_state_dict(_state_dict['model_state'])
        self.optimizer.load_state_dict(_state_dict['optimizer_state'])
        self.optimizer.name = _state_dict['optimizer_name']
        self.model.name = _state_dict['optimizer_name']
        
        print(f"LOADING MODEL, epoch {_state_dict['epoch']}"
                 + f", time {_state_dict['time']}")
        

    def freeze_unfreeze_model(self, freeze: bool = True):
        """
        Function which freezes and unfreezes models.

        Args:
            * model, pytorch model
            * freeze, bool, True for freeze and False for unfreeze
        """
        # Check if resnet
        if self.model_params.name in ['res', 'incept_v3']:
            _seq = True
        else:
            _seq = False

        # Freeze or unfreeze
        freeze_model_base(self.model, freeze = freeze, seq = _seq)

        # Notice
        if freeze == True:
            print(f"USING MODEL: {self.model_params.name}, WEIGHTS: FROZEN")
        else:
            print(f"USING MODEL: {self.model_params.name}, WEIGHTS: UNFROZEN")

        # Refresh optimizer settings
        self.optimizer = self.init_optimizer()    

        # Send it to gpu  
        if self.model_params.gpu != False and self.model_params.gpu != 'cpu':
            print(f"USING GPU: {self.device}")
            self.model = self.model.to(self.device)
        else:
            print("USING CPU")

        print("**************************************")


    def transfer_weights(self, path:str):
        """
        Function which transfer all weights from model given by the path to the current model. Weights must be valid to be transfered.

        Args:
            * path, str, path to model which transfer weights
        """

        # Decoration
        print("**************************************")
        print("TRANSFERING WEIGHTS...")

        transfer_weights_to_model(path, self.model)

        # Decorative
        print("**************************************")

    #*******************************************************#
    # Optimzer
    #*******************************************************#  
    def init_optimizer(self):
        """
            Init optimizer: Feel free to add other optmizers. Learnign rate is important.
            Must be called last (everything else must be intialised for find lr to work)
            https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py
        """
        # Calculate learning rate if necessary
        if self.model_params.learning_rate == "Auto":
            

            _model = lr_wrap_model(self.aug_model, self.model)
            print("Finding best learning rate...")
            if self.model_params.loss_name == 'c_entropy':
                _criterion = torch.nn.CrossEntropyLoss()
            
            if self.model_params.opt_name == 'ADAM':
                _opt = torch.optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=1e-3)
            if self.model_params.opt_name == 'ADAMW':
                _opt = torch.optim.AdamW(self.model.parameters(), lr=1e-7, weight_decay=1e-3)
            
            _lr_finder = LRFinder(_model, _opt, _criterion, device=self.device)
            _lr_finder.range_test(self.train_dl, end_lr=100, num_iter=100)
            _q = _lr_finder.plot() # to inspect the loss-learning rate graph            
            self.model_params.learning_rate = _q[1]
            _lr_finder.reset()
        print(self.model_params.learning_rate)
        
        print(f"USING OPTIMIZER: {self.model_params.opt_name} / LR:{self.model_params.learning_rate}")
        assert self.model_params.opt_name in ["ADAM", "ADAMW"], f"Wrong optimizer name, got: {self.model_params.opt_name}"

        if self.model_params.opt_name == 'ADAM':
            return torch.optim.Adam(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'ADAMW':
            return torch.optim.AdamW(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
    #*******************************************************#
    # Loss function
    #*******************************************************#
    def init_loss(self):
        """
        Init loss function: Feel free to add other loss functions.
        """
        if self.model_params.loss_name == 'c_entropy':
            print(f"USING LOSS FUNCTION: {self.model_params.loss_name}")
            return torch.nn.CrossEntropyLoss(reduction='none')
        
        if self.model_params.loss_name == 'mse':
            print(f"USING LOSS FUNCTION: {self.model_params.loss_name}")
            return torch.nn.MSELoss(reduction='none')
    
    
    #*******************************************************#
    # Training subrutine
    #*******************************************************#
    def train_one_epoch(self, data):
        """
        Training model function. 

        Args:
            * data, dataloader of the train dataset
        """

        # Storage for metrics calculation
        _loss_storage = torch.zeros(len(data), 1, device = self.device)
        _mae_storage = torch.zeros(len(data), 1, device = self.device)

        # Swap to mode train
        self.model.train()

        # Shuffle dataset and create enum object
        data.dataset.shuffle_samples()
        _batch_iter = enumerate(data)
        
        # Go trough batches
        for _index, _batch in _batch_iter:

            # Clear grads
            self.optimizer.zero_grad()

            # Calc loss
            if self.model_params.loss_name == 'c_entropy':
                _loss = self.get_c_entropy_loss(_index, _batch, _predictions, _true, True)

            if self.model_params.loss_name == 'mse':
                _loss, _mae = self.get_mse_loss(_index, _batch, True)
                
            
            # Propagate loss
            _loss.backward()

            # Apply loss
            self.optimizer.step()

            # Save loss
            _loss_storage[_index] = _loss.detach()
            _mae_storage[_index] = _mae.detach()
        
        # Return metrics
        return _loss_storage, _mae_storage

    
    #*******************************************************#
    # Validation subrutine
    #*******************************************************#
    def validate_model(self, data):
        """
        Validation model function

        Args:
            * data, dataloader of the train dataset
        """

        # # Storage for metrics calculation
        _loss_storage = torch.zeros(len(data), 1, device = self.device)
        _mae_storage = torch.zeros(len(data), 1, device = self.device)

        # We don't need calculate gradients 
        with torch.no_grad():
            # Set model in evaluate mode - no batchnorm and dropout
            self.model.eval()

            # Go trough data
            for _index, _batch in enumerate(data):
                # Get loss
                if self.model_params.loss_name == 'c_entropy':
                    _loss = self.get_c_entropy_loss(_index, _batch, _predictions, _true, False)

                if self.model_params.loss_name == 'mse':
                    _loss, _mae = self.get_mse_loss(_index, _batch, False)
                    
                # Save loss
                _loss_storage[_index] = _loss.detach()
                _mae_storage[_index] = _mae.detach()
        
        # Return staff
        return _loss_storage, _mae_storage



    #*******************************************************#
    # Crossentropy loss
    #*******************************************************#
    def get_c_entropy_loss(self, index, batch, predictions, true, augumentation = True):
        """
        Function that calculates cross entropy loss. Loss in this code is MeanSquaredError

        Args:
            * index, int, batch index needed to populate _metrics

            * batch, tensor, data

            * _predictions, _true, lists, lists to store predictions and true values of batch

            * augmentation, boolean, True if augumentation is to be applied
        """

        # Parse _batch
        _image, _label, _id, _examID, _modality, _bpe = batch
        
        # Transfer data to GPU
        _input_data = _image.to(self.device, non_blocking = True)
        _output_data = _label.to(self.device, non_blocking = True)

        # Augment data
        if self.aug_model != None and augumentation == True:
            _input_data = self.aug_model(_input_data)
        
        # If augumentation is required only for color scheme (validation set)
        if self.aug_model != None and augumentation == False:
            if 'RGB' in self.model_params.augumentation_model:
                _input_data = self.aug_model_color_only(_input_data)
            else:
                _input_data = self.aug_model_color_only(_input_data)


        # Caluclate loss
        if self.model_params.name == 'incept_v3' and self.model.training:
            # Inception has auxuliry outputs...
            _prediction, _aux_output = self.model(_input_data)
        else:
            _prediction = self.model(_input_data)
            
        _loss = self.loss(_prediction, _output_data)

        # Detach from graph
        _prediction = _prediction.detach()
        _output_data = _output_data.detach()


        # For metrics
        with torch.no_grad():
            _prediction_arg_max = torch.argmax(_prediction, dim = 1)

            # Fix last batch size
            if _prediction_arg_max.shape[0] != self.train_dl.batch_size:
                _prediction_arg_max = torch.cat((_prediction_arg_max, torch.zeros([self.train_dl.batch_size - _prediction_arg_max.shape[0]], device = self.device)))
            
            if _output_data.shape[0] != self.train_dl.batch_size:
                _output_data = torch.cat((_output_data, torch.zeros([self.train_dl.batch_size - _output_data.shape[0]], device = self.device)))

            # Save
            predictions[index] = _prediction_arg_max
            true[index] = _output_data
        
        # Return mean of all loss          
        return _loss.mean()
    

    #*******************************************************#
    # MSE loss
    #*******************************************************#
    def get_mse_loss(self, index, batch, augumentation = True):
        """
        Function that calculates cross entropy loss. Loss in this code is MeanSquaredError

        Args:
            * index, int, batch index needed to populate _metrics

            * batch, tensor, data

            * _predictions, _true, lists, lists to store predictions and true values of batch

            * augmentation, boolean, True if augumentation is to be applied
        """

        # Parse _batch
        _image, _label, _id, _examID, _modality, _bpe = batch
        
        # Transfer data to GPU
        _input_data_image = _image.to(self.device, non_blocking = True)
        
        if self.aug_model == None:
            _input_data = _input_data_image
        
        # Augment data
        if self.aug_model != None and augumentation == True:
            _input_data = self.aug_model(_input_data_image)
        
        # If augumentation is required only for color scheme (validation set)
        if self.aug_model != None and augumentation == False:
            if 'RGB' in self.model_params.augumentation_model:
                _input_data = self.aug_model_color_only(_input_data_image)
            else:
                _input_data = self.aug_model_color_only(_input_data_image)


        # Caluclate loss
        _prediction = self.model(_input_data)
        _loss = self.loss(_prediction, _input_data)
        _mae = torch.nn.functional.l1_loss(_prediction, _input_data)

        # Return mean of all loss          
        return _loss.mean(), _mae

    #*******************************************************#
    # Function for evaluation, 
    #*******************************************************#
    def eval_metrics(self, epoch, loss, mae, mode,save_dict)->float:
        """
            Function for metric evaluation

        Args:
            * epoch, int, epoch number
            * mae, torch.tensor, mae values
            * loss, torch.tensor, loss values
            * mode, str, 'valid', 'train', 'test' - just cosmetic
            * save_dict, dict, dictionary for metrics update

        Return:
            * Calculated loss over complete dataset
        
        """ 
        # Print info
        print("{}, {}, Loss:{:.5f}".format(mode, epoch, torch.mean(loss.double())))
        print("{}, {}, MAE:{:.5f}".format(mode, epoch, torch.mean(mae.double())))
        
        # Save to dict
        save_dict['epoch'].append(epoch)
        save_dict['loss'].append(torch.mean(loss.double()))
        save_dict['mae'].append(torch.mean(mae.double()))
        
                
        # Return loss value for model evaluation, or any other score.
        return torch.mean(loss.double())
    
    #*******************************************************#
    # Function for model saving
    #*******************************************************#
    def save_model(self, epoch, best, info):
        """
            Function for model saving

            Args:
                * epoch, int, epoch being saved
                * best, boolean, Is this the best model
                * info, str, decoration for the name
        """

        _name = f"{self.model_params.name}_{self.model_params.opt_name}:" +  \
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" + \
                    f"_{epoch}"

        _model = self.model
        
        # For paralel
        if isinstance(_model, torch.nn.DataParallel):
            _model = _model.module
        
        # Define saving state
        _state = {
            'time': str(datetime.datetime.now()),
            'model_state': _model.state_dict(),
            'model_name': type(_model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }
        
        # Save last model
        torch.save(_state, self.results_output_dir+_name + '.pth')
        print('Saving model!')
        
        # Save best model
        if best:
            print('Saving best model!')
            _name = f"{self.model_params.name}_{self.model_params.opt_name}_" +  \
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" 
            torch.save(_state, self.results_output_dir + _name + f'{info}_best_model.pth')

    #*******************************************************#
    # Function for model loading
    #*******************************************************#

    def load_model(self, path):
        """
            Function that loads model.

            Args:
                * path, string, path to the model checkpoint
        """
        print("LOADING MODEL")
        
        _state_dict = torch.load(path)
        self.model.load_state_dict(_state_dict['model_state'])
        self.optimizer.load_state_dict(_state_dict['optimizer_state'])
        self.optimizer.name = _state_dict['optimizer_name']
        self.model.name = _state_dict['optimizer_name']
        
        print(f"LOADING MODEL, epoch {_state_dict['epoch']}"
                 + f", time {_state_dict['time']}")
        
    #*******************************************************#
    # Export metrics to xlsx
    #*******************************************************#
    def export_metrics_to_xlsx(self, best_epoch, best_score, training_dict, validation_dict):
        """
        Function that exports model's training and validation metrics to dictionary
        """

        # Transfer data to cpu - loss only
        _loss =[] 
        for _item in training_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        training_dict['loss'] = _loss    

        _mae =[] 
        for _item in training_dict['mae']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _mae.append(_item)
        training_dict['mae'] = _mae
        
        _loss = []
        for _item in validation_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        validation_dict['loss'] = _loss   

        _mae =[] 
        for _item in validation_dict['mae']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _mae.append(_item)
        validation_dict['mae'] = _mae 

        # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir+ f"{self.model_params.name}_{self.model_params.opt_name}_" +  
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}" + 
                    f"_{best_epoch}" + f"{best_score:5f}.xlsx", engine = 'xlsxwriter')

        # Generate dataframes
        _df_train = pd.DataFrame.from_dict(training_dict)
        _df_valid = pd.DataFrame.from_dict(validation_dict)

        _df_train.to_excel(_writer, sheet_name="Training", index = False)
        _df_valid.to_excel(_writer, sheet_name="Validation", index = False)
        _writer.close() 

    #*******************************************************#
    # Function for predicting on set
    #*******************************************************#
    def model_predict_from_dl(self, input_data_loader, save_name:str):
        """
        Function is predicting results for the given dataloader and stores them
        
        Input args:
            * input_data_loader, pytorch dataloader, dataloader for which the results are going to be predicted
            * save_name, str, name of the file where the results are going to be stored
        """

        # Notice
        print("######################################")
        print(f"Predicting results on the dataloader")

        # Storage
        _results_dict = {
            'loss' : [], 
            'mae' : []
        }

        # Obtain results
        _loss, _mae = self.validate_model(input_data_loader)
        
        # Evaluate metrics
        # Transfer to cpu

        _loss_l =[] 
        for _item in _loss:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss_l.append(_item)
        _results_dict['loss'] = _loss_l    

        _mae_l =[] 
        for _item in _mae:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _mae_l.append(_item)
        _results_dict['mae'] = _mae_l

         # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir+ f"{self.model_params.name}_{self.model_params.opt_name}_" +  
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}_" + 
                    save_name+ ".xlsx", engine = 'xlsxwriter')

        # Generate dataframes
        _df_report = pd.DataFrame()
        _df_report['MSE'] = np.mean(_results_dict['loss'])
        _df_report['MAE'] = np.mean(_results_dict['mae'])
        _df_predictions = pd.DataFrame()
        _df_predictions['MSE'] = _results_dict['loss']
        _df_predictions['MAE'] = _results_dict['mae']

        # Export
        _df_report.to_excel(_writer, sheet_name="Report", index = False)
        _df_predictions.to_excel(_writer, sheet_name="Predictions", index = False)
        _writer.close() 

    def __start_training_with_interupts(self):
        """
        Fork of start training utilized for sake of code clearance. It is easer to 
        copy paste and control everything from main train function than adapt each function
        to support validation checking after each % of training images seen
        """

        print(f"TRAINING STARTED: Epochs: {self.model_params.epochs}, Validation after: {self.model_params.valid_epochs}")

        # Set dictionaries for metrics
        _training_results_dict = {
            'epoch' : [], 'loss' : [], 
            'mae': []
            }

        _valid_results_dict = {
            'epoch' : [], 'loss' : [], 
            'mae': []
            }
        
        # Validation params
        _mod_valid = int(self.model_params.valid_epochs.split('_')[0])
        _threshold_valid = int(self.model_params.valid_epochs.split('_')[1])


        # Set score 
        _best_loss = 1000.0
        _best_epoch = 0

        # Storage for metrics calculation

        _loss_storage = torch.zeros(len(self.train_dl), 1, device = self.device)
        _mae_storage = torch.zeros(len(self.train_dl), 1, device = self.device)

        # Start training
        for _epoch in range(1, self.model_params.epochs +1):
            print("--------------------------------------")
            print(f"Epoch {_epoch} / {self.model_params.epochs}")
            
            # Run training function
            _start = time.time()
            
            # Swap to mode train
            self.model.train()
            
            # Shuffle dataset and create enum object
            self.train_dl.dataset.shuffle_samples()
            
            _batch_iter = enumerate(self.train_dl)
            
            # Calculate percentage indexes
            _number_of_batches = len(self.train_dl)
            
            _percentage = self.model_params.valid_percent
            
            # Go trough batches
            for _index, _batch in _batch_iter:
                # Clear grads
                self.optimizer.zero_grad()
                # Calc loss
                if self.model_params.loss_name == 'c_entropy':
                    _loss = self.get_c_entropy_loss(_index, _batch, _predictions, _true, True)
                
                if self.model_params.loss_name == 'mse':
                    _loss, _mae = self.get_mse_loss(_index, _batch, True)

                # Propagate loss
                _loss.backward()
                
                # Apply loss
                self.optimizer.step()

                # Storage loss
                _loss_storage[_index] = _loss.detach()
                _mae_storage[_index] = _mae.detach()


                if _index >=_percentage * _number_of_batches:
                    print("######################################")
                    print(f"Evaluation at percentage: {_percentage}")
                    _percentage += self.model_params.valid_percent
                    
                    # Print train
                    print("Train, Loss:{:.5f}".format(_loss))
                    print("Train, MAE:{:.5f}".format(_mae))
                    
                    # Storage
                    _valid_interim_results_dict = {
                    'epoch' : [], 'loss' : [], 
                    'mae': []}

                    _loss_v, _mae_v = self.validate_model(self.valid_dl)
                    _epoch_loss = self.eval_metrics(_epoch, _loss_v, _mae_v, 
                                            'valid', _valid_interim_results_dict)

                    # Save best model
                    if _epoch_loss < _best_loss:
                        self.save_model(_epoch, best = True, info = 'interim')
                        _best_loss = _epoch_loss
                        _best_epoch = _epoch

                    # Report to wandb
                    if self.model_params.wandb == True:
                        _validation_metrices = {'batch': _index,
                                    'Valid-interim-loss': _valid_interim_results_dict['loss'][-1],
                                    'Valid-interim-mae': _valid_interim_results_dict['mae'][-1],
                        }

                        # Wandb report
                        if self.model_params.wandb == True:
                            wandb.log(_validation_metrices)

                    # Swap to mode train
                    self.model.train()

            print("######################################")                   
            # Run evaluation function
            _epoch_loss = self.eval_metrics(_epoch, _loss_storage, _mae_storage, 
                                            'train', _training_results_dict)



            # Report to Wandb
            if self.model_params.wandb == True:
                _wandb_metrices = {'epoch': _epoch,
                                'Train-loss': _epoch_loss,
                                'Train-mae': _training_results_dict['mae'][-1],
                }
        
            # Save model
            if _epoch % self.model_params.save_epochs == 0:
                self.save_model(_epoch, best = False, info = 'train')

            # Validation
            print("######################################")
            if _epoch == 1 or _epoch % _mod_valid == 0 or _epoch >= _threshold_valid:
                _loss_v, _mae_v = self.validate_model(self.valid_dl)
                _epoch_loss = self.eval_metrics(_epoch, _loss_v, _mae_v, 
                                            'valid', _valid_results_dict)
            
                # Save best model
                if _epoch_loss < _best_loss:
                    self.save_model(_epoch, best = True, info = 'valid')
                    _best_loss = _epoch_loss
                    _best_epoch = _epoch

                # Report to wandb
                if self.model_params.wandb == True:
                    _validation_metrices = {'epoch': _epoch,
                                'Valid-loss': _valid_results_dict['loss'][-1],
                                'Valid-mae': _valid_results_dict['mae'][-1],
                    }
                    _wandb_metrices.update(_validation_metrices)
             
                
            # Wandb report
            if self.model_params.wandb == True:
                wandb.log(_wandb_metrices)
            
            # Report time
            _end = time.time()
            print(f"Time: {(_end-_start):5f}sec")
            
            # Cosmetic
            print("######################################")

            # Early stopping
            if _best_epoch + self.model_params.early_stopping <= _epoch:
                print(f"Early stopping at epoch: {_epoch}")
                break
        
        # Save metrics
        self.export_metrics_to_xlsx(_best_epoch, _best_loss, 
                            _training_results_dict, _valid_results_dict)

        # Release memory
        torch.cuda.empty_cache() 

        # Finish wandb
        if self.model_params.wandb == True:
            wandb.finish()


    #*******************************************************#
    # Main function for training
    #*******************************************************#
    def start_training(self):
        """
        Function which controls training of the model
        """

        # If valid_percent is on, the training forks to another main rutine due to the
        # code smootheness
        if self.model_params.valid_percent != None:
            self.__start_training_with_interupts()
            return ()

        print(f"TRAINING STARTED: Epochs: {self.model_params.epochs}, Validation after: {self.model_params.valid_epochs}")

        # Set dictionaries for metrics
        _training_results_dict = {
            'epoch' : [], 'loss' : [], 
            'mae': []
            }

        _valid_results_dict = {
            'epoch' : [], 'loss' : [], 
            'mae': []
            }
        
        # Set score 
        _best_loss = 1000.0
        _best_epoch = 0

        # Validation params
        _mod_valid = int(self.model_params.valid_epochs.split('_')[0])
        _threshold_valid = int(self.model_params.valid_epochs.split('_')[1])

        # Start training
        for _epoch in range(1, self.model_params.epochs +1):
            print("--------------------------------------")
            print(f"Epoch {_epoch} / {self.model_params.epochs}")
            
            # Run training function
            _start = time.time()
            _loss, _mae = self.train_one_epoch(self.train_dl)
            _end = time.time()
            
            # Run evaluation function
            _epoch_loss = self.eval_metrics(_epoch, _loss, _mae, 
                                            'train', _training_results_dict)
            
            # Report to Wandb
            if self.model_params.wandb == True:
                _wandb_metrices = {'Epoch': _epoch,
                                'Train-loss': _training_results_dict['loss'][-1],
                                'Train-mae': _training_results_dict['mae'][-1]
                }
            
            # Report time
            print(f"Time: {(_end-_start):5f}sec")
            # Save model
            if _epoch % self.model_params.save_epochs == 0:
                self.save_model(_epoch, best = False)
            
            # Validation
            print("######################################")
            if _epoch == 1 or _epoch % _mod_valid == 0 or _epoch >= _threshold_valid:
                _loss_v, _mae_v = self.validate_model(self.valid_dl)
                _epoch_loss = self.eval_metrics(_epoch, _loss_v, _mae_v, 
                                            'valid', _valid_results_dict)
                # Save best model
                if _epoch_loss < _best_loss:
                    self.save_model(_epoch, best = True)
                    _best_loss = _epoch_loss
                    _best_epoch = _epoch

                # Report to wandb
                if self.model_params.wandb == True:
                    _validation_metrices = {'Epoch': _epoch,
                                'Valid-loss': _valid_results_dict['loss'][-1],
                                'Valid-mae': _valid_results_dict['mae'][-1]
                    }
                    _wandb_metrices.update(_validation_metrices)
          
                
            # Wandb report
            if self.model_params.wandb == True:
                wandb.log(_wandb_metrices)

            # Cosmetic
            print("######################################")

            # Early stopping
            if _best_epoch + self.model_params.early_stopping <= _epoch:
                print(f"Early stopping at epoch: {_epoch}")
                break
        
        # Save metrics
        self.export_metrics_to_xlsx(_best_epoch, _best_loss, 
                            _training_results_dict, _valid_results_dict)

        # Release memory
        torch.cuda.empty_cache() 

        # Finish wandb
        if self.model_params.wandb == True:
            wandb.finish()

        

