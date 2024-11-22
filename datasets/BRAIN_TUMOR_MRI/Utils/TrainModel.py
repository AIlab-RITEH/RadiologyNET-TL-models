from collections import namedtuple
import torch
from Utils.models import *
import numpy as np
import datetime
import pandas as pd
import xlsxwriter
from sklearn import metrics
import time
import shutil
import os
from Utils.dataloader_BrainTumorMRI import *
from Utils.config import LoaderConfig, ModelConfig
from torch_lr_finder import LRFinder

class model_training_app:
    
    def __init__(self, train_dl, valid_dl, model_params, results_output_dir):
        """
        init training with given params
        
        Args:
            * train_dl, train data set dataloader
            * valid_dl, validation data set dataloader
            * model_params, name+model params.
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
        self.model_params: ModelConfig = model_params

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

        print("**************************************")

    #*******************************************************#
    # Init augumentation model
    #*******************************************************# 
    def init_augumentation_model(self):
        """
        Script which inits augumentation model
        """
        if self.model_params.augumentation_model == 'RGB_transform':
            self.aug_model = TransformToRGB()
            self.aug_model_color_only = TransformToRGB()          
        

        if self.model_params.augumentation_model == 'GRAY_transform':
            self.aug_model = AddChanels()
            self.aug_model_color_only = AddChanels()
            
        if self.model_params.augumentation_model == 'BRAIN_RGB':
            self.aug_model = TransformRGB_BRAIN()
            self.aug_model_color_only = TransformToRGB()      
            
        if self.model_params.augumentation_model == 'BRAIN_GRAY':
            self.aug_model = TransformGray_BRAIN()
            self.aug_model_color_only = AddChanels()  
        

    #*******************************************************#
    # Model handling scripts
    #*******************************************************#   
    def init_model(self):
        """
        Model is initially unfrozen
        """
        print(f"USING MODEL: {self.model_params.name}, WEIGHTS: UNFROZEN")

        if self.model_params.name == 'dense121':
            _model = DenseNet121(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        elif self.model_params.name == 'eff3':
            _model = EfficientNetB3(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        elif self.model_params.name == 'eff4':
            _model = EfficientNetB4(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        elif self.model_params.name == 'inceptionV3':
            _model = InceptionV3(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)

        elif self.model_params.name == 'res50':
            _model = ResNet50(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)

        elif self.model_params.name == 'res18':
            _model = ResNet18(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)

        elif self.model_params.name == 'res34':
            _model = ResNet34(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = True)
        
        elif self.model_params.name == 'mobileNetV3Small':
            _model = MobileNetV3Small(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        elif self.model_params.name == 'mobileNetV3Large':
            _model = MobileNetV3Large(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        elif self.model_params.name == 'vgg16':
            _model = VGG16(pretrained = self.model_params.pretrained, number_of_classes = self.model_params.number_of_output_classes)
            freeze_model_base(_model, freeze = False, seq = False)

        else:
            raise NotImplementedError(f'Model backbone {self.model_params.name} not implemented')

        # Send it to gpu
        if self.model_params.gpu != False and self.model_params.gpu != "cpu":
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            print("USING CPU")

        return _model        

    def freeze_unfreeze_model(self, freeze: bool = True):
        """
        Function which freezes and unfreezes model weights.

        Args:
            * model, pytorch model
            * freeze, bool, True for freeze and False for unfreeze
        """
        # Check if resnet
        if self.model_params.name in ['res18','res34','res50','res101', 'res152', 'incept_v3', 'inceptionV3']:
            _seq = True
        elif self.model_params.name in ['eff0', 'eff1', 'eff2', 'eff3', 'eff4', 'mobileNetV3Small', 'mobileNetV3Large', 'vgg16', 'dense121']:
            _seq = False
        elif self.model_params.name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']:
            _seq = 'vit'
        else:
            raise NotImplementedError(f'freeze_unfreeze_model :: setting seq :: not implemented for {self.model_params.name}')


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
        Function which transfers all weights from model given by the path to the current model.

        Args:
            * path, str, path to model which transfer weights
        """

        # Decoration
        print("**************************************")
        print("TRANSFERING WEIGHTS...")

        transfer_weights_to_model(path, self.model, device = self.device)

        # Decorative
        print("**************************************")

    #*******************************************************#
    # Optimzer
    #*******************************************************#  
    def init_optimizer(self):
        """
            Init optimizer: Feel free to add other optmizers. Learning rate is important.
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
            if self.model_params.opt_name == 'SGD':
                _opt = torch.optim.SGD(self.model.parameters(), lr = 1e-7)
            
            _lr_finder = LRFinder(_model, _opt, _criterion, device=self.device)
            _lr_finder.range_test(self.train_dl, end_lr=100, num_iter=100)
            _q = _lr_finder.plot() # to inspect the loss-learning rate graph            
            self.model_params.learning_rate = _q[1]
            _lr_finder.reset()
        print(self.model_params.learning_rate)
        
        print(f"USING OPTIMIZER: {self.model_params.opt_name} / LR:{self.model_params.learning_rate}")
        assert self.model_params.opt_name in ["ADAM", "ADAMW", "SGD"], f"Wrong optimizer name, got: {self.model_params.opt_name}"

        if self.model_params.opt_name == 'ADAM':
            return torch.optim.Adam(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'ADAMW':
            return torch.optim.AdamW(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(),  lr = self.model_params.learning_rate)
        
    #*******************************************************#
    # Loss function
    #*******************************************************#
    def init_loss(self):
        """
        Init loss function: Feel free to add other loss functions.
        """
        print(f"USING LOSS FUNCTION: {self.model_params.loss_name}")
        return torch.nn.CrossEntropyLoss(reduction='none')
    
    
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
        _predictions = torch.zeros(len(data.dataset), device = self.device)
        _true = torch.zeros(len(data.dataset), device = self.device)
        _loss_storage = torch.zeros(len(data), device = self.device)
        
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
            _loss = self.get_c_entropy_loss(_index, _batch, _predictions, _true, True)
                
            # Propagate loss
            _loss.backward()

            # Apply loss
            self.optimizer.step()

            # Save loss
            _loss_storage[_index] = _loss.detach()
        
        # Return metrics
        return _predictions, _true, _loss_storage

    
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
        # Storage for metrics calculation
        _predictions = torch.zeros(len(data.dataset), device = self.device)
        _true = torch.zeros(len(data.dataset), device = self.device)
        _loss_storage = torch.zeros(len(data), device = self.device)

        # We don't need calculate gradients 
        with torch.no_grad():
            # Set model in evaluate mode - no batchnorm and dropout
            self.model.eval()

            # Go trough data
            for _index, _batch in enumerate(data):
                # Get loss
                _loss = self.get_c_entropy_loss(_index, _batch, _predictions, _true, False)
                
                # Save loss
                _loss_storage[_index] = _loss.detach()
        
        # Return staff
        return _predictions, _true, _loss_storage

    #*******************************************************#
    # Crossentropy loss
    #*******************************************************#
    def get_c_entropy_loss(self, index, batch, predictions, true, augumentation = True):
        """
        Function that calculates cross entropy loss.

        Args:
            * index, int, batch index needed to populate _metrics

            * batch, tensor, data

            * _predictions, _true, lists, lists to store predictions and true values of batch

            * augmentation, boolean, True if augumentation is to be applied. augmentation should not be applied
                during validation - but should be applied during training.
        """

        # Parse _batch
        _image, _label, _label_str, _path = batch
        
        # Transfer data to GPU
        _input_data = _image.to(self.device, non_blocking = True)
        _output_data = _label.to(self.device, non_blocking = True)

        # Augment data
        if self.aug_model != None and augumentation == True:
            _input_data = self.aug_model(_input_data)
        
        # If augumentation is required only for color scheme (validation set)
        if self.aug_model != None and augumentation == False:
            _input_data = self.aug_model_color_only(_input_data)

        # Caluclate loss
        _prediction = self.model(_input_data)	
        _loss = self.loss(_prediction, _output_data)

        # Detach from graph
        _prediction = _prediction.detach()
        _output_data = _output_data.detach()

        # For metrics
        with torch.no_grad():
            _prediction_arg_max = torch.argmax(_prediction, dim = 1)

            # Fix last batch size.
            predictions[index * self.train_dl.batch_size: index * self.train_dl.batch_size + _prediction.shape[0] ] = _prediction_arg_max
            true[index * self.train_dl.batch_size: index * self.train_dl.batch_size + _output_data.shape[0] ] = _output_data
        
        # Return mean of all loss          
        return _loss.mean()

    #*******************************************************#
    # Function for evaluation, 
    #*******************************************************#
    def eval_metrics(self, epoch, predictions, true, loss, mode ,save_dict)->float:
        """
            Function for metric evaluation

        Args:
            * epoch, int, epoch number
            * predictions, torch.tensor, prediction for each batch
            * true, torch.tensor, true value for each batch
            * loss, torch.tensor, loss values
            * mode, str, 'valid', 'train', 'test' - just cosmetic
            * save_dict, dict, dictionary for metrics update

        Return:
            * Calculated loss over complete dataset
        
        """
        # Transfer to cpu
        _predictions = predictions.to('cpu')
        _true = true.to('cpu')

        # Get 1D array
        _predictions = torch.flatten(_predictions)
        _true = torch.flatten(_true)

        
        # Calculate precission recall f1 score
        _report = metrics.classification_report(_predictions, _true, digits=3, zero_division = 0, output_dict = True)
        _weighted_report = _report['weighted avg']
        _macro_report = _report['macro avg']
 
        # Print info
        print("{}, {}, Loss:{:.3f}".format(mode, epoch, torch.mean(loss.double())))
        print("{}, {}, Weighted : Precision:{:.3f}, Recall:{:.3f} F1-score: {:.3f}".format(mode, epoch, 
                                                                                 _weighted_report['precision'],
                                                                                 _weighted_report['recall'],
                                                                                 _weighted_report['f1-score']))
        print("{}, {}, Macro : Precision:{:.3f}, Recall:{:.3f} F1-score: {:.3f}".format(mode, epoch, 
                                                                                 _macro_report['precision'],
                                                                                 _macro_report['recall'],
                                                                                 _macro_report['f1-score']))
        
        # Save to dict
        save_dict['epoch'].append(epoch)
        save_dict['loss'].append(torch.mean(loss.double()))
        save_dict['precision'].append(_macro_report['precision'])
        save_dict['recall'].append(_macro_report['recall'])
        save_dict['f1-score'].append(_macro_report['f1-score'])
        save_dict['precision-w'].append(_weighted_report['precision'])
        save_dict['recall-w'].append(_weighted_report['recall'])
        save_dict['f1-score-w'].append(_weighted_report['f1-score'])

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
        _name = f"last_model_checkpoint"  # use this to conserve memory

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
        if best == False:
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
        Function that exports model's training and validation metrics to a dictionary
        """

        # Transfer data to cpu - loss only
        _loss =[] 
        for _item in training_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        training_dict['loss'] = _loss              
        _loss =[] 
        for _item in validation_dict['loss']:
            _item = _item.to('cpu')
            _item = _item.detach().numpy()
            _loss.append(_item)
        validation_dict['loss'] = _loss   

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
    # Main function for training
    #*******************************************************#
    def start_training(self):
        """
        Function which controls training of the model
        """

        print(f"TRAINING STARTED: Epochs: {self.model_params.epochs}, Validation after: {self.model_params.valid_epochs}")

        # Set dictionaries for metrics
        _training_results_dict = {
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
        }

        _valid_results_dict = {
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
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
            _predictions, _true, _loss = self.train_one_epoch(self.train_dl)
            _end = time.time()
            
            # Run evaluation function
            _epoch_loss = self.eval_metrics(_epoch, _predictions, _true, 
                                            _loss, 'train', _training_results_dict)
                        
            # Report time
            print(f"Time: {(_end-_start):5f}sec")
            # Save model
            if _epoch % self.model_params.save_epochs == 0:
                self.save_model(_epoch, best = False, info = 'train')
            
            # Validation
            print("######################################")
            if _epoch == 1 or _epoch % _mod_valid == 0 or _epoch >= _threshold_valid:
                _predictions_v, _true_v, _loss_v = self.validate_model(self.valid_dl)
                _epoch_loss = self.eval_metrics(_epoch, _predictions_v, _true_v, 
                                            _loss_v, 'valid', _valid_results_dict)
                # Save best model
                if _epoch_loss < _best_loss:
                    self.save_model(_epoch, best = True, info = 'valid')
                    _best_loss = _epoch_loss
                    _best_epoch = _epoch

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
            'epoch' : [], 'loss' : [], 
            'precision' : [], 'recall' : [], 
            'f1-score' : [], 'precision-w' : [], 
            'recall-w' : [],  'f1-score-w' : [],
        }

        # Obtain results
        _predictions, _true, _loss = self.validate_model(input_data_loader)
        
        # Evaluate metrics
         # Transfer to cpu
        _predictions = _predictions.to('cpu')
        _true = _true.to('cpu')

        # Get 1D array
        _predictions = torch.flatten(_predictions)
        _true = torch.flatten(_true)

        # Calculate precission recall f1 score
        _report = metrics.classification_report(_predictions, _true, digits=3, zero_division = 0, output_dict = True)

         # Generate writer for a given model      
        _writer = pd.ExcelWriter(self.results_output_dir+ f"{self.model_params.name}_{self.model_params.opt_name}_" +  
                    f"{self.model_params.learning_rate}" + f"_{self.model_params.loss_name}_" + 
                    save_name+ ".xlsx", engine = 'xlsxwriter')

        # Generate dataframes
        _df_report = pd.DataFrame(_report ).transpose()
        _df_predictions = pd.DataFrame()
        _df_predictions['True'] = _true
        _df_predictions['Predicted'] = _predictions

        # Export
        _df_report.to_excel(_writer, sheet_name="Report", index = False)
        _df_predictions.to_excel(_writer, sheet_name="Predictions", index = False)
        _writer.close() 