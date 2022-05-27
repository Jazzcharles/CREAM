## CREAM: Weakly Supervised Object Localization via Class RE-Activation Mapping
This is the official PyTorch implementation for CREAM.
We provide the inference code for regressor-based CREAM on CUB dataset. 
The training code is coming soon.
### Requirements
`pip install -r requirements.txt`


### Prepare Data
Download the official Caltech-UCSD-Birds-200-2011 (CUB) dataset. 


### Inference
`mkdir checkpoints`

Download the pretrained VGG16/InceptionV3 models for CUB and put them under checkpoints/:
- https://drive.google.com/drive/folders/1pgLjFN_LxKEA7dnmyVORUxFoEi_k9q86?usp=sharing

This example shows the usage of inference with vgg16 classification and localization model.

`bash infer.sh`
    
This inference code is based on PSOL (https://github.com/tzzcl/PSOL).
