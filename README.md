# EfficienPS-lit

This project is a PyTorchLightning attempt at implementing EfficientPS: https://arxiv.org/pdf/2004.02307.pdf
This project is in progress and does not reproduce the original paper results.
Takes up to 25Gb of memory for a batch of 1. 

The repository still needs support for:
* Batch higher than 1
* Similar data augmentation descibed on the training protocol section (pre trained EfficientNet on image etc...)
* Different datasets

## Tested on
Pytorch-lightning 1.0.3, torch 1.6.0, torchvision 0.7.0.

### Built and spcialized for cityscape:
You will need cityscape fine annotation in a data/ folder
