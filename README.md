# RadiologyNET Foundation Models

Welcome to the offical repository of RadiologyNET foundation models.


[RadiologyNET](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00373-1) is a large dataset consisting of ~1.3 mil. DICOM files, mostly chest/head MR and CT images. This dataset was used to pretrain several popular architectures that were evaluated on several publicly available datasets, which cover a range of anatomical regions (hand, wrist, head, lungs, chest) and imaging modalities (MR, CR, CT). The models were pretrained using [PyTorch](https://pytorch.org/).



## Evaluation
These models were evaluated on five publicly available datasets, against those pretrained on ImageNet and those pretrained on randomly initialised weights (i.e. *Baseline*).

When resources (i.e. training time and training data) were not restriced, the obtained metrics of the three approaches did not differ statistically. However, we found that RadiologyNET models could prove beneficial in resource-limited conditions, and could give a boost in the performance during the first few epochs of training. 

![Brain Tumor MRI - Performance in first 10 epochs of training](assets/BRAIN_TUMOR__AVERAGE_training_progress_IN_FIRST_10_EPOCHS__valid_set__['res50',%20'mobileNetV3Small']__partitions-[1].png)

**Fig.1**: Average performance of best-performing models across first 10 epochs on the Brain Tumor MRI dataset. Results per each epoch are averaged across five runs.



![COVID-19 - Performance in first 10 epochs of training](assets/COVID19__AVERAGE_training_progress_IN_FIRST_10_EPOCHS__valid_set__['res18',%20'mobileNetV3Large']__partitions-[1].png)

**Fig.2**: Average performance of best-performing models across first 10 epochs on the COVID-19 dataset. Results per each epoch are averaged across five runs.


## Usage

The `Example.ipynb` notebook in the `./usage/` directory demonstrates how to load weights into your model.
We have prepared a function called `transfer_weights_to_model()` which searches layer namespace and transfers the weights. The benefits of using this function over just loading the features is that this function will transfer weights even if the network topologies do not match exactly, or if the desired outcome is to load the weights partially (for example, loading ResNet50 into a U-Net-ResNet50 topology).

Loading the RadiologyNET weights can be performed in three steps. For example, if using MobileNetV3Large:

```python
import models

RADIOLOGYNET_WEIGHTS = 'TL_Weights/MobileNetV3Large.pth'
model = models.MobileNetV3Large(pretrained=False, number_of_classes=NUM_CLASSES)
models.transfer_weights_to_model(path=RADIOLOGYNET_WEIGHTS, target_model=model, device='cpu')
```

#### Step 1
Download the RadiologyNET weights and place them into the `./usage/TL_Weights/` directory. The download links can be found [here](#download).
```python
RADIOLOGYNET_WEIGHTS = 'TL_Weights/MobileNetV3Large.pth'  # path to the downloaded RadiologyNET weights
```

#### Step 2
Instantiate your desired model. The `./usage/models.py` script contains the topologies tested with RadiologyNET. For example, if loading MobileNetV3Large is desired:

```python
import models  # loads the models.py script found in the ./usage/ directory
model = models.MobileNetV3Large(pretrained=False, number_of_classes=10)  # instantiate MobileNetV3Large from the implementation in models.py
```

#### Step 3
Transfer the weights.

```python
models.transfer_weights_to_model(path=RADIOLOGYNET_WEIGHTS, target_model=model, device='cpu')
```

For more information, refer to the `./usage/` directory.


<a id="download"></a>

## Download

The models (pytorch weights) are packaged into `.tar.gz` archives and are available at the following download links:

| Model            | File Size |                                                                         |
|------------------|-----------|-------------------------------------------------------------------------|
| DenseNet121      | 74 MiB     | [Download](http://radiologynet.riteh.hr/models/DenseNet121.tar.gz)      |
| EfficientNetB3   | 114 MiB    | [Download](http://radiologynet.riteh.hr/models/EfficientNetB3.tar.gz)   |
| EfficientNetB4   | 189 MiB    | [Download](http://radiologynet.riteh.hr/models/EfficientNetB4.tar.gz)   |
| InceptionV3      | 229 MiB    | [Download](http://radiologynet.riteh.hr/models/InceptionV3.tar.gz)      |
| MobileNetV3Large | 45 MiB     | [Download](http://radiologynet.riteh.hr/models/MobileNetV3Large.tar.gz) |
| MobileNetV3Small | 17 MiB     | [Download](http://radiologynet.riteh.hr/models/MobileNetV3Small.tar.gz) |
| ResNet18         | 117 MiB    | [Download](http://radiologynet.riteh.hr/models/ResNet18.tar.gz)         |
| ResNet34         | 223 MiB    | [Download](http://radiologynet.riteh.hr/models/ResNet34.tar.gz)         |
| ResNet50         | 247 MiB    | [Download](http://radiologynet.riteh.hr/models/ResNet50.tar.gz)         |
| VGG16            | 1.21 GiB   | [Download](http://radiologynet.riteh.hr/models/VGG16.tar.gz)            |


## Challenges
The following challenges have been used to evaluate the performance of RadiologyNET foundation models. Brain Tumor MRI is described in detail in the `./datasets/` folder, with a MinimalWorkingSample.ipynb available for easier reproducibility.

* [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)
* [GRAZPEDWRI-DX](https://www.nature.com/articles/s41597-022-01328-z)
* [RSNA Pediatric Bone Age Challenge 2017](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
* [LUng Nodule Analysis Challenge](https://luna16.grand-challenge.org/Data/)
