# Encapsulate Capsule Layers


## prequisite
- Python=3.9
- CUDA: 11.2/11.3
- Pytorch framwork: 1.12.1, pytorch-lightning
- Others: numpy, opencv, scipy
- dashboard: neptune ai (for training), gradio (for testing)
## Environments Settings
- Install [Anaconda](https://www.anaconda.com/)
- create a new environment:
```
conda create --name=Capsule python=3.9
conda activate Capsule
```
- Install dependencies: 
```
sh env-create.sh
```
#
## Environments Settings
- Run: 
```
python Capsule/train.py -c config/new-config.yml
```

## Configuration (Config file)
The configurations, a [config.yaml](Modules/config.yaml), encompassing the model architecture and training settings, as well as dataset settings. The "config.yaml" file follows a structured format, consisting of a list of dictionaries. Each dictionary within the list represents a distinct configuration and saves specific configuration parameters.

<table>
<tr>
<td colspan=1>
    Table 2. Configuration
</td>

| Parameters  | Description |Scope | Value |
| ------------- | ------------- | ------------- | ------------- |
| `logger` | neptune account |  |  |
| project | your project | logger |str  |
| api_key | your account token | logger |str  |
| tags | Runtime Tags | logger |[str]  |
| task | task of experiment |  | str: "classification", "detection", "segmentation" |
| `Model` |
| name | Model's name  | architect_settings  | string |
| name | Pretrained model  | architect_settings/backbone  | string: "name"-"s/m/l" |
| is_full | If True, use full model  | architect_settings/backbone  | Bool |
| is_pretrained |  pretrained weights  | architect_settings/backbone  | Bool |
| is_freeze | Freeze weights  | architect_settings/backbone  | Bool |
| n_cls | num classes  | architect_settings | int |
| `Dataset` |
| name | Dataset name  | dataset_settings | string: "LungCT-Scan", "CIFAR10", "PennFudan" |
| path | path to dataset  | dataset_settings  | string |
| img_size | size of image to model  | dataset_settings  | int |
| `Training` |
| gpu_ids | list of gpus used  | training_settings  | list: [0] |
| n_gpu | num gpus  | training_settings  | int |
| img_size | size of image to model  | training_settings  | int |
| loss | loss function  | training_settings  | str: "ce" (classification/segmentation), "dice", "mse", "none"(detection) |
| metric | metric name  | training_settings  | str: "accuracy", "dice", "mAP" |
| ckpt_path | path to check-points  | training_settings  | str |
| n_epoch | num epoch  | training_settings  | int |
| n_batch | batch size  | training_settings  | int |
| num_workers | num workers to dataloader | training_settings  | int |
| optimizer | optimizer | training_settings  | str: "adam", "sgd" |
| lr_scheduler | learning rate scheduler | training_settings  | str: "step", "multistep", "reduce_on_plateau" |
| early_stopping | early stopping | training_settings  | bool |
| lr | learning rate | training_settings  | float|
| lr_step | learning rate step for decay| training_settings  | int|
| lr_decay | learning rate decay rate | training_settings  | float|
| momentum | momentum for optimizer | training_settings  | float|
| weight_decay | weight decay for "sgd" | training_settings  | float|