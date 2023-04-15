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
We reimplement Capsule Layers in 3 files: Routing.py, CapsuleLayer.py, and Model.py
- Routing.py: Implement 3 routing methods: [EM](Capsules/Routing.py#L23), [Dynamic](Capsules/Routing.py#84), and [Fuzzy](Capsules/Routing.py#L126). Algorithm's details are provided in [pdf](Algorithm.pdf)
- CapsuleLayer.py: Implement [2D ConvCaps](Capsules/CapsuleLayer.py#L70), [Primary Capsules](Capsules/CapsuleLayer.py#L29), [Shortcut Layers](Capsules/CapsuleLayer.py#L260), and [Efficient DepthWise Capsule](Capsules/CapsuleLayer.py#L319).

![alt text](image/4.png)
![alt text](image/5.png)
- Model.py: Using above implemented modules to build a [CapsuleNetwork](Capsules/Model.py#L97), [Shortcut Architecture](Capsules/Model.py#L181), and [Efficient CapsuleNetwork](Capsules/Model.py#L254) (recommended).

#
## Training Interface
Examples of training Capsule Network can be found in [ReadDataset.py](Capsules/ReadDataset.py) and [main.py](Capsules/main.py), we config hyper-parameters in [config.yaml](Capsules/config.yaml) file

- ReadDataset.py: there are five datasets have been pre-defined: Mnist, [affNist](Capsules/ReadDataset.py#L153), Fashion Mnist, SVHN, and [smallNorb](Capsules/ReadDataset.py#L14)
- main.py: Our main module is [CapsuleModel](Capsules/main.py#L80) that based on [pytorch-lightning](https://lightning.ai/pages/open-source/) and logged by [neptune-ai](https://neptune.ai/)

![alt text](image/run.png)

```
python Capsules/main.py
```
#

## Testing Interface
We deploy (demo) our model using [Gradio](https://gradio.app/), which supports  activation map visualization and see the results
```
python Capsules/Interface.py
```
![alt text](image/gradio.png)

#
## Configuration (Config file)
We set up the configurations (model achitecture and training settings) using a [config.yaml](Capsules/config.yaml). The config file is a list of dictionaries, where a dictionary saves a specific configuration. The structure is as follows:
```
"name": {
    "architect_settings": {
        "model": one of architectures in ["eff", "base", "convolution", "shortcut"] (recommended "eff")
        "reconstructed": using reconstructed loss (recommended True)
        ... (model configurationm, number of layers, and details of each layer, it depends on task, input and output size)
        "routing": {
            "type": routing method, ["dynamic", "em", "fuzzy"]
            "params": parameters for routing method
                      - for dynamic routing: [iterations] (e.x [3])
                      - for em routing: [iterations, beta] (e.x [3, 0.01])
                      - for fuzzy routing: [iterations, beta, m] (e.x [3, 0.01, 2])
        }
    }
    "training_settings":{
        "loss": we prepare 3 loss for classication ["ce", "margin", "spread"] and a loss for segmentation ["bce"]
        "CAM": set Fasle when training and True for drawing Class Activation map in demo.
        ... (other training parameters such as batch size, epoch, learning rate)
        "dataset": path to the training dataset, we prepared some datasets ["affNist","CenterMnist", "smallNorb"] for classification and ["CT-scan-dataset"] for segementation, the dataset shoud be inside the "data" directory.
        "ckpt_path": path to checkpoint
    }
}
```
Please refer to the config file to understand how it works, and please understand that to ensure the flexibility of the module we can not further hard coding the parameters because `tunning the parameters` is an important part of the development. We plan to continue optimizing the configurations so the config file may be updated in the future.

#
## list of papers
- [EM routing](https://openreview.net/pdf?id=HJWLfGWRb)
- [Dynamic routing](https://arxiv.org/pdf/1710.09829.pdf)
- [Shortcut routing](https://search.ieice.org/bin/pdf_link.php?category=A&fname=e104-a_8_1043&lang=E&year=2021)

## list of implementation
- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)

- [danielhavir/capsule-network](https://github.com/danielhavir/capsule-network)

- [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)

- [lidq92/pytorch-capsule-networks](https://github.com/lidq92/pytorch-capsule-networks)

- [jindongwang/Pytorch-CapsuleNet](https://github.com/jindongwang/Pytorch-CapsuleNet)