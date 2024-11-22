# Brain Tumor MRI

This directory contains the scripts used to evaulate the performance of ImageNet, RadiologyNET and models trained *from Scratch* (i.e. *Baseline*). The minimal working sample for reproducing our evaluation is presented here in detail.

## Dataset

The Brain Tumor MRI dataset is available [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The downloaded data should contain two directories: `Training/` and `Testing/`. Place them both into the `Data/` directory.

## Weights And TL

For RadiologyNET TL models, download the desired weights and place them into the `TL_Weights` directory, then extract them to get the `.pth` files. The `TL_Weights` directory should look something like this:

```
TL_Weights/
    DenseNet121.pth
    MobileNetV3Small.pth
    ResNet50.pth
    [...]
    VGG16.pth
```

The `TrainRadiologyNet.py` script contains the entire code necessary to transfer the weights onto the downstream task models. Particularly, the `transfer_weights_to_model()` method found in `Utils/models.py`  does exactly this: transfer the pretrained RadiologyNET weights onto the newly instantiated models, by searching the layer namespace (for info can be found in `../usage/`).

ImageNet weights are automatically downloaded from [torch repositories](https://pytorch.org/vision/stable/models.html).

## Training and Evaluation
The parameters and training configs are located in three scripts:
* `TrainRadiologyNet.py`
* `TrainImageNet.py`
* `TrainFromScratch.py`

The configuration files can be adjusted as needed. The current settings are optimised for a minimal working sample with reasonable training times. While Brain Tumor MRI models were originally trained for 200 epochs with early stopping (with a 10-epoch early stoppying mechanism), in the current settings - this was reduced to a maximum of 10 epochs for brevity. Additionally, multiple learning rates were explored in the original study, but the current settings use a fixed learning rate of $10^-3$ (also for simplicity).

Also, all of the provided weights are set to train in the specified scripts. One can pick-and-choose which topology should be trained and tested by changing the `for` loop in each of the three scripts. For example, changing the code to this:
```
for backbone in [
        'mobileNetV3Small',
        'res50',
]:
    # [ training code... ]
```
will only train the MobileNetV3Small and ResNet50 topologies.

The trained models will be saved to the `MODELS/` directory. The results of the evaluation (`EvaluateBrain.py`) will be stored to a directory called `Results_Train_Valid_Test/` which will be created once `EvaluateBrain.py` is executed.

## Run minimal working sample
Once the requirements were installed using the provided `requirements.txt` file; and the RadiologyNET weights and the Brain Tumor MRI dataset have been downloaded, then the training and inference scripts can be run using the `MinimalWorkingSample.ipynb`. 

## Codecarbon
The training process also includes the `codecarbon` package, the docs can be found [here](https://mlco2.github.io/codecarbon/examples.html). The results are stored into an `emissions.csv` file.
