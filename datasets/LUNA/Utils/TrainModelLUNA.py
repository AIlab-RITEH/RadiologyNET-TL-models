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
import wandb

from torch_lr_finder import LRFinder

METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3
METRICS_SIZE = 5


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
        
        # Init prescale model
        self.prescale_model = self.init_prescale_model()
        self.prescale_model = self.prescale_model.to(self.device)

        # Augumentation model
        self.aug_model = None
        if self.model_params.augumentation_model != None:
            self.init_augumentation_model()           

        # Loss function
        self.loss = self.init_loss

        # Init optimzer (must be last for lr find)
        self.optimizer = self.init_optimizer()
        
   
        # Wandb
        if self.model_params.wandb == True:
            self.init_wandb()

        print("**************************************")
    
    #*******************************************************#
    # Init augumentation model
    #*******************************************************# 
    def init_prescale_model(self):
        """
        Script which initialise prescaling model
        """
        if self.model_params.augumentation_model == 'LUNA_RGB':
            return LUNA_RGB()    
        if self.model_params.augumentation_model == 'LUNA_GRAY':
            return LUNA_GRAY()
    
    #*******************************************************#
    # Init augumentation model
    #*******************************************************# 
    def init_augumentation_model(self):
        """
        Script which inits augumentation model
        """

        _aug_dict = {}
        _aug_dict['flip'] = self.model_params.flip
        _aug_dict['offset'] = self.model_params.offset
        _aug_dict['scale'] = self.model_params.scale
        _aug_dict['rotate'] = self.model_params.rotate
        _aug_dict['noise'] = self.model_params.noise
        self.aug_model = SegmentationAugmentation(**_aug_dict)
        self.aug_model = self.aug_model.to(self.device)
        return
                
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
        project=f"{self.model_params.custom_info}" + f"{self.model_params.name}_"+f"{_current_time}",
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
                          
        if self.model_params.name == 'UNET':
            _model = UNet(weights_path = self.model_params.weights_path, 
                          pretrained = self.model_params.pretrained, 
                          n_classes = self.model_params.number_of_labels,
                          backbone = self.model_params.backbone)
        # Send it to gpu
        if self.model_params.gpu != False and self.model_params.gpu != "cpu":
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            print("USING CPU")

        return _model
   
    def freeze_unfreeze_model(self, freeze: bool = True):
        """
        Function which freezes and unfreezes models.

        Args:
            * model, pytorch model
            * freeze, bool, True for freeze and False for unfreeze
        """
        # Freeze or unfreeze
        freeze_model_base(self.model, freeze = freeze)

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

        if self.model_params.name == 'UNET':
            _model = UNet(weights_path = path, 
                          pretrained = self.model_params.pretrained, 
                          n_classes = self.model_params.number_of_labels,
                          backbone = self.model_params.backbone)
        # Send it to gpu
        if self.model_params.gpu != False and self.model_params.gpu != "cpu":
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            print("USING CPU")
	
	# Create model
        self.model = _model
        
        # Create optimizer
        self.optimizer = self.init_optimizer()

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
         
        print(f"USING OPTIMIZER: {self.model_params.opt_name} / LR:{self.model_params.learning_rate}")

        if self.model_params.opt_name == 'ADAM':
            return torch.optim.Adam(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
        if self.model_params.opt_name == 'ADAMW':
            return torch.optim.AdamW(self.model.parameters(), lr = self.model_params.learning_rate, weight_decay=1e-3)
        
    #*******************************************************#
    # Loss function
    #*******************************************************#
    def init_loss(self, prediction_g, label_g, epsilon=1):
        """
        Init loss function: This is dice loss sepcially tailored for this particular issue.
        Args:
            * prediction_g, tensor, model's prediction
            * label_g, tensor, ground truth
            * epsilon, number, small nomber to avoid division by 0)
        """
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g
    
    #*******************************************************#
    # Batch loss
    #*******************************************************#
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        """
        Middlemad function for batch loss computation
        
        Args:
            * batch_ndx, index of the patch being trained.
            * batch_tup, tensor, batch
            * batch_size, integer, number of samples in batch
            * metrices_g, storage dict, place to storage caluclated metrics
            * classificationThreshold, float 0-1, classification threshold
        """
        # Dismantel batch
        _input_t = batch_tup['ct']
        _label_t = batch_tup['pos']
        _series_list = batch_tup['series_uid']
        _slice_ndx_list = batch_tup['slice_index']


        # Transfer data to gpu/cpu devices
        _input_g = _input_t.to(self.device, non_blocking=True)
        _label_g = _label_t.to(self.device, non_blocking=True)

        # Augument data if neccessary
        if self.model.training and self.aug_model != None:

            _input_g, _label_g = self.aug_model(_input_g, _label_g)
        
        # Prescaling
        _input_g = self.prescale_model(_input_g)

        # Predict
        _prediction_g = self.model(_input_g)
        
        # Obtain loss 
        _diceLoss_g = self.loss(_prediction_g, _label_g)
        _fnLoss_g = self.loss(_prediction_g * _label_g, _label_g)

        # To save metrices
        _start_ndx = batch_ndx * batch_size
        _end_ndx = _start_ndx + _input_t.size(0)

        with torch.no_grad():
            _predictionBool_g = (_prediction_g[:, 0:1]
                                > 0.5).to(torch.float32)

            _tp = (     _predictionBool_g *  _label_g).sum(dim=[1,2,3])
            _fn = ((1 - _predictionBool_g) *  _label_g).sum(dim=[1,2,3])
            _fp = torch.logical_xor(_predictionBool_g, _label_g).sum(dim=[1,2,3])

            metrics_g[METRICS_LOSS_NDX, _start_ndx:_end_ndx] = _diceLoss_g
            metrics_g[METRICS_TP_NDX, _start_ndx:_end_ndx] = _tp
            metrics_g[METRICS_FN_NDX, _start_ndx:_end_ndx] = _fn
            metrics_g[METRICS_FP_NDX, _start_ndx:_end_ndx] = _fp

        return _diceLoss_g.mean() + _fnLoss_g.mean() * 8
    
    #*******************************************************#
    # Training subrutine
    #*******************************************************#
    def train_one_epoch(self, data):
        """
        Training model function. 

        Args:
            * data, dataloader of the train dataset
        
        """

        # Create storage
        _trnMetrics_g = torch.zeros(METRICS_SIZE, len(self.train_dl.dataset), device=self.device)
        
        # Swap to mode train
        self.model.train()

        # Shuffle dataset and create enum object
        data.dataset.shuffleSamples()
        _batch_iter = enumerate(data)

        for _batch_ndx, _batch_tup in _batch_iter:
            self.optimizer.zero_grad()

            # Obtain loss
            _loss_var = self.computeBatchLoss(_batch_ndx, _batch_tup, data.batch_size, _trnMetrics_g)

            # Backpropagate loss            
            _loss_var.backward()
            
            # Applay it
            self.optimizer.step()

        return _trnMetrics_g.to('cpu')

    
    #*******************************************************#
    # Validation subrutine
    #*******************************************************#
    def validate_model(self, data):
        """
        Validation model function

        Args:
            * data, dataloader of the train dataset
        """
        # We don't need calculate gradients 
        with torch.no_grad():
            # Set model in evaluate mode - no batchnorm and dropout
            self.model.eval()
            
            # Storage for metrics calculation
            _valMetrics_g = torch.zeros(METRICS_SIZE, len(self.valid_dl.dataset), device=self.device)
            
            # Go trough data
            for _batch_ndx, _batch_tup in enumerate(data):
            
             # Obtain loss
             _loss_var = self.computeBatchLoss(_batch_ndx, _batch_tup, data.batch_size, _valMetrics_g)
        # Return staff
        return _valMetrics_g.to('cpu')

    #*******************************************************#
    # Function for evaluation, 
    #*******************************************************#
    def eval_metrics(self, epoch_ndx, mode_str, metrics_t, save_dict)->float:
        """
            Function for metric evaluation

        Args:
            * epoch_ndx, int, epoch number
            * metrc_t, tenost dict, metrics to safe
            * mode_str, str, 'valid', 'train', 'test' - just cosmetic
            * save_dict, dict, something to save things

        Return:
            * Calculated loss over complete dataset
        
        """ 
        # Extract metrics
        _metrics_a = metrics_t.detach().numpy()
        _sum_a = _metrics_a.sum(axis=1)
        # IDK
        assert np.isfinite(_metrics_a).all()
        
        # Count number of labels
        _allLabel_count = _sum_a[METRICS_TP_NDX] + _sum_a[METRICS_FN_NDX]
        
        # Create storage dict
        _metrics_dict = {}
        _metrics_dict['loss/all'] = _metrics_a[METRICS_LOSS_NDX].mean()

        # Calculate TP,FN,FN
        _metrics_dict['percent_all/tp'] = \
            _sum_a[METRICS_TP_NDX] / (_allLabel_count or 1) * 100
        _metrics_dict['percent_all/fn'] = \
            _sum_a[METRICS_FN_NDX] / (_allLabel_count or 1) * 100
        _metrics_dict['percent_all/fp'] = \
            _sum_a[METRICS_FP_NDX] / (_allLabel_count or 1) * 100

        # Get precision and recal
        _precision = _metrics_dict['pr/precision'] = _sum_a[METRICS_TP_NDX] \
            / ((_sum_a[METRICS_TP_NDX] + _sum_a[METRICS_FP_NDX]) or 1)
        _recall    = _metrics_dict['pr/recall']    = _sum_a[METRICS_TP_NDX] \
            / ((_sum_a[METRICS_TP_NDX] + _sum_a[METRICS_FN_NDX]) or 1)

        # Calculate f1
        _metrics_dict['pr/f1_score'] = 2 * (_precision * _recall) \
            / ((_precision + _recall) or 1)

         # Print info
        print(("E{} {:8} " + "{loss/all:.4f} loss, " + "{pr/precision:.4f} precision, " + "{pr/recall:.4f} recall, " \
             + "{pr/f1_score:.4f} f1 score").format(epoch_ndx, mode_str, **_metrics_dict))     
        # Save to dict
        save_dict['epoch'].append(epoch_ndx)
        save_dict['loss'].append(_metrics_a[METRICS_LOSS_NDX].mean())
        save_dict['TP'].append(_metrics_dict['percent_all/tp'])
        save_dict['FN'].append(_metrics_dict['percent_all/fn'])
        save_dict['FP'].append(_metrics_dict['percent_all/fp'])
        save_dict['Precision'].append(_precision)
        save_dict['Recall'].append(_recall)
        save_dict['F1Score'].append(_metrics_dict['pr/f1_score'])
        
        return _metrics_dict['pr/recall']
    
    #**********************************as*********************#
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
            _loss.append(_item)
        training_dict['loss'] = _loss    

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
        import matplotlib.pyplot as plt
        # Notice
        print("######################################")
        print(f"Predicting results on the dataloader")
        
        # Create results dir
        

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            shutil.rmtree(save_name)           # Removes all the subdirectories!
            os.makedirs(save_name)
            
        self.model.eval()
        for batch_tup in input_data_loader:
         
            _input_t = batch_tup['ct']
            _label_t = batch_tup['pos']
            _series_list = batch_tup['series_uid']
            _slice_ndx_list = batch_tup['slice_index']
            
            # Transfer data to gpu/cpu devices
            _input_g = _input_t.to(self.device, non_blocking=True)

            # Prescaling
            _input_g = self.prescale_model(_input_g)

            # Predict
            with torch.no_grad():
                _prediction_g = self.model(_input_g)
            
            for _img, _label, _prediction, _series, _slice in zip(_input_t, _label_t, _prediction_g, _series_list, _slice_ndx_list):
                #print(_img)
                _slice = str(_slice.tolist())

                


                _name = _series+"_"+_slice
                
                _path = os.path.join(save_name,_name)
                if not os.path.exists(_path):
                    os.makedirs(_path)
                else:
                    _path+=_slice
                    os.mkdir(_path)

                # Save input
                _img = _img.squeeze()
                _img = _img.cpu().numpy()
                plt.figure()
                plt.imshow(_img, cmap = 'gray')
                plt.axis('off')  # Disabling di axis
                plt.savefig(os.path.join(_path,"original.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
		# Save mask
		 # Save input
                _img = _label.squeeze()
                _img = _img.cpu().numpy()
                plt.figure()
                plt.imshow(_img, cmap = 'gray')
                plt.axis('off')  # Disabling di axis
                plt.savefig(os.path.join(_path,"GT_mask.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
                		# Save mask
		 # Save prediction
                _img = _prediction.squeeze()
                _img = _img.cpu().numpy()
                plt.figure()
                plt.imshow(_img, cmap = 'gray')
                plt.axis('off')  # Disabling di axis
                plt.savefig(os.path.join(_path,"PRED_mask.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
                # Then, convert the tensor to a numpy array
                #image_array = tensor.cpu().numpy().transpose(1, 2, 0)  # Assuming tensor is in NCHW format
                
                # Convert numpy array to PIL Image
                #image = Image.fromarray((image_array * 255).astype('uint8'))
                # Save the PIL Image
                #image.save('tensor_image.jpg')


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
            'TP': [], 'FN': [], 'FP': [],
            'Precision': [], 'Recall': [], 'F1Score': []
            }

        _valid_results_dict = {
            'epoch' : [], 'loss' : [], 
            'TP': [], 'FN': [], 'FP': [],
            'Precision': [], 'Recall': [], 'F1Score': []
            }
        
        # Set score 
        _best_score = 0.0
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
            _trnMetrics_t= self.train_one_epoch(self.train_dl)
            _end = time.time()
            
            # Run evaluation function
            _epoch_score = self.eval_metrics(_epoch, 'train', _trnMetrics_t, _training_results_dict)
            
            # Report to Wandb
            if self.model_params.wandb == True:
                _wandb_metrices = {'Epoch': _epoch,
                                'Train-loss': _training_results_dict['loss'][-1],
                                'Train-TP': _training_results_dict['TP'][-1],
                                'Train-FN': _training_results_dict['FN'][-1],
                                'Train-FP': _training_results_dict['FP'][-1],
                                'Train-precision': _training_results_dict['Precision'][-1],
                                'Train-recall': _training_results_dict['Recall'][-1],
                                'Train-f1score': _training_results_dict['F1Score'][-1],
                }
            
            # Report time
            print(f"Time: {(_end-_start):5f}sec")
            # Save model
            if _epoch % self.model_params.save_epochs == 0:
                self.save_model(_epoch, best = False, info = "train")
            
            # Validation
            print("######################################")
            if _epoch == 1 or _epoch % _mod_valid == 0 or _epoch >= _threshold_valid:
                _valMetrics_t = self.validate_model(self.valid_dl)
                _epoch_score = self.eval_metrics(_epoch,'valid', _valMetrics_t, _valid_results_dict)
                
                # Save best model
                if _epoch_score > _best_score:
                    self.save_model(_epoch, best = True, info = "valid")
                    _best_score = _epoch_score
                    _best_epoch = _epoch

                # Report to wandb
                if self.model_params.wandb == True:
                    _validation_metrices = {'Epoch': _epoch,
                                'Valid-loss': _valid_results_dict['loss'][-1],
                                'Valid-TP': _valid_results_dict['TP'][-1],
                                'Valid-FN': _valid_results_dict['FN'][-1],
                                'Valid-FP': _valid_results_dict['FP'][-1],
                                'Valid-precision': _valid_results_dict['Precision'][-1],
                                'Valid-recall': _valid_results_dict['Recall'][-1],
                                'Valid-f1score': _valid_results_dict['F1Score'][-1],
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
        self.export_metrics_to_xlsx(_best_epoch, _best_score, 
                            _training_results_dict, _valid_results_dict)

        # Release memory
        torch.cuda.empty_cache() 

        # Finish wandb
        if self.model_params.wandb == True:
            wandb.finish()
