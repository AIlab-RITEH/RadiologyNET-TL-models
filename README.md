
<div align="center">
  <img src="assets/radiologynet_logo.jpg" alt="RadiologyNET Logo"/>
  <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
  <img src="assets/ai_lab.png" alt="RITEH AI Lab Logo"/>
</div>


# RadiologyNET Foundation Models

Welcome to the official repository of RadiologyNET foundation models.  

[RadiologyNET](https://www.nature.com/articles/s41598-025-05009-w) is a large-scale, [pseudo-labelled](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00373-1) medical imaging dataset, comprising over 1.9 million DICOM-derived images spanning various anatomical regions and imaging modalities (MR, CT, CR, RF, XA). The dataset was used to pretrain several widely used neural network architectures, which were then evaluated across multiple downstream tasks. The models were pretrained using [PyTorch](https://pytorch.org/).

These pretrained models are made publicly available to support further research and development in medical transfer learning.

### Contents:
1. [Evaluation and Findings](#evaluation)
2. [Usage](#usage)
3. [Download](#download)
4. [Tested Challenges](#challenges)
5. [Notes & Limitations](#notes)
6. [Citation](#citation)

---

<a id="evaluation"></a>
## Evaluation and Findings

RadiologyNET models were benchmarked against ImageNet-pretrained and randomly initialised (Baseline) models on five publicly available medical datasets. Key findings:

- When training resources were unrestricted, RadiologyNET and ImageNet models achieved comparable performance. 
- RadiologyNET showed advantages under resource-limited conditions (e.g., early training stages, small datasets).
- Multi-modality pretraining generally yielded better generalisation than single-modality alternatives, but this depended on the intra-domain variability of each modality. Where a single modality was sufficiently diverse, there was no significant benefit from incorporating other modalities into the pretraining dataset.
- High-quality manual labelling (e.g. [RadImageNet](https://www.radimagenet.com/)) remains the gold standard.

![Brain Tumor MRI](assets/BRAIN_TUMOR__AVERAGE_training_progress_IN_FIRST_10_EPOCHS__valid_set__['res50',%20'mobileNetV3Small']__partitions-[1].png)
**Fig.1**: Validation performance during the first 10 epochs on Brain Tumor MRI. Averaged across five runs.

![COVID-19](assets/COVID19__AVERAGE_training_progress_IN_FIRST_10_EPOCHS__valid_set__['res18',%20'mobileNetV3Large']__partitions-[1].png)
**Fig.2**: Validation performance during the first 10 epochs on COVID-19 dataset. Averaged across five runs.

---

<a id="usage"></a>
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


<a id="challenges"></a>
## Challenges
The following challenges have been used to evaluate the performance of RadiologyNET foundation models. Brain Tumor MRI is described in detail in the `./datasets/` folder, with a MinimalWorkingSample.ipynb available for easier reproducibility.

* [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)
* [GRAZPEDWRI-DX](https://www.nature.com/articles/s41597-022-01328-z)
* [RSNA Pediatric Bone Age Challenge 2017](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
* [LUng Nodule Analysis Challenge](https://luna16.grand-challenge.org/Data/)

<a id="notes"></a>
#### Notes & Limitations
- RadiologyNET labels were generated using an unsupervised clustering of image, DICOM, and diagnosis text features.
- The current version is based on a single-institution dataset. Contributions of multi-centre datasets may be included in future iterations.
- For more details, we refer the reader to the [full publication](https://doi.org/10.1038/s41598-025-05009-w) published in *Scientific Reports*.

<a id="citation"></a>
## Citation
If you use these models, please cite (BibTeX):
``` 
@article{Napravnik2025,
  title = {Lessons learned from RadiologyNET foundation models for transfer learning in medical radiology},
  volume = {15},
  ISSN = {2045-2322},
  url = {http://dx.doi.org/10.1038/s41598-025-05009-w},
  DOI = {10.1038/s41598-025-05009-w},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Napravnik,  Mateja and Hržić,  Franko and Urschler,  Martin and Miletić,  Damir and Štajduhar,  Ivan},
  year = {2025},
  month = jul 
}

@article{Napravnik2024,
  title = {Building RadiologyNET: an unsupervised approach to annotating a large-scale multimodal medical database},
  volume = {17},
  ISSN = {1756-0381},
  url = {http://dx.doi.org/10.1186/s13040-024-00373-1},
  DOI = {10.1186/s13040-024-00373-1},
  number = {1},
  journal = {BioData Mining},
  publisher = {Springer Science and Business Media LLC},
  author = {Napravnik,  Mateja and Hržić,  Franko and Tschauner,  Sebastian and Štajduhar,  Ivan},
  year = {2024},
  month = jul 
}
```

> 📄 **Reference**:  
> Napravnik M, Hržić F, Urschler M, Miletić D, Štajduhar I.  
> *Lessons learned from RadiologyNET foundation models for transfer learning in medical radiology*.  
> Scientific Reports 15, 21622 (2025).  
> https://doi.org/10.1038/s41598-025-05009-w
