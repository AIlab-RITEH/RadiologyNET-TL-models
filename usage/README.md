# How to use RadiologyNET weights

The `Example.ipynb` notebook demonstrates how to load weights into your model.
We have prepared a function called `transfer_weights_to_model()` which searches layer namespace and transfers the weights. The benefits of using this function over just loading the features is that this function will transfer weights even if the network topologies do not match exactly, or if the desired outcome is to load the weights partially (for example, loading ResNet50 into a U-Net-ResNet50 topology).


Loading the RadiologyNET weights can be performed in three steps. For example, if using MobileNetV3Large:

```python
import models

RADIOLOGYNET_WEIGHTS = 'TL_Weights/MobileNetV3Large.pth'
model = models.MobileNetV3Large(pretrained=False, number_of_classes=NUM_CLASSES)
models.transfer_weights_to_model(path=RADIOLOGYNET_WEIGHTS, target_model=model, device='cpu')
```

### Step 1
Download the RadiologyNET weights and place them into the `TL_Weights/` directory.
```python
RADIOLOGYNET_WEIGHTS = 'TL_Weights/MobileNetV3Large.pth'  # path to the downloaded RadiologyNET weights
```

### Step 2
Instantiate your desired model. The `models.py` script contains the topologies tested with RadiologyNET. For example, if loading MobileNetV3Large is desired:

```python
import models  # loads the models.py script found in this directory
model = models.MobileNetV3Large(pretrained=False, number_of_classes=10)  # instantiate MobileNetV3Large from the implementation in models.py
```

### Step 3
Transfer the weights.

```python
models.transfer_weights_to_model(path=RADIOLOGYNET_WEIGHTS, target_model=model, device='cpu')
```

## Transferring RadiologyNET weights

The full implementation of `transfer_weights_to_model()` is given below. For more information, see the `models.py` implementation.


```python
def transfer_weights_to_model(path: str, target_model, device = 'cpu'):
    _src_state_dict = torch.load(path, map_location = device)
    _target_state_dict = target_model.state_dict()
    
    for _src_name, _src_param in _src_state_dict['model_state'].items():
        
        if _src_name in _target_state_dict:
            if _src_param.shape == _target_state_dict[_src_name].shape:
                _target_state_dict[_src_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")
        else:
            print(f"LAYER NOT FOUND: {_src_name}")
        
        _expanded_name = "model."+_src_name
        print(f"TRYING with expanded name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")
        
        _expanded_name = _src_name.replace("features", "model")
        print(f"TRYING with swapped name: {_expanded_name}")
        if _expanded_name in _target_state_dict:
            if _src_param.shape == _target_state_dict[_expanded_name].shape:
                _target_state_dict[_expanded_name].copy_(_src_param)
                continue
            else:
                print(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")

        if len(_src_name.split("model.")) == 1:
            continue

        _shorted_name = _src_name.split("model.")[1]
        print(f"TRYING with shortened name: {_shorted_name}")
        if _shorted_name in _target_state_dict:
            if _src_param.shape == _target_state_dict[_shorted_name].shape:
                _target_state_dict[_shorted_name].copy_(_src_param)
            else:
                print(f"UNABLE TO TRANSFER at layer: {_src_name}/{_src_param.shape}")
        else:
            print(f"LAYER NOT FOUND: {_src_name}")

    # Update weights
    target_model.load_state_dict(_target_state_dict)

```