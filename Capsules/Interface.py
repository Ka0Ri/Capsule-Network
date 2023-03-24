import gradio as gr
import os
import yaml
import torch
from main import CapsuleModel
from ReadDataset import affNistread
from torch.utils.data import DataLoader
import random
from pytorch_lightning import LightningModule, Trainer


with open("Capsules/config.yaml", 'r') as stream:
    try:
        PARAMS = yaml.safe_load(stream)
       
    except yaml.YAMLError as exc:
        print(exc)

# model = CapsuleModel(PARAMS)
convmodel = CapsuleModel.load_from_checkpoint("model/affNist-conv/epoch=193-step=15326.ckpt", PARAMS=PARAMS["gradio"])
convmodel.eval()
capsule = CapsuleModel.load_from_checkpoint("model/affNist-fuzzy-shortcut/epoch=129-step=10270.ckpt", PARAMS=PARAMS["gradio1"])
# disable randomness, dropout, etc...
capsule.eval()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

from torchvision import transforms
Train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
            
Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

Test_data = affNistread(mode="test", data_path="affMnist", transform=Test_transform)


# trainer = Trainer(accelerator="gpu", devices=[1])


def load_data():
    index = random.randint(0, 1000)
    x, y = Test_data[index]
    img = transforms.ToPILImage()(x)
    dummy = torch.zeros_like(x)
    x_batch = torch.stack([x, dummy])
    cls = convmodel(x_batch)[0]
    cls = torch.softmax(cls, dim=-1).tolist()
    labels = {k: float(v) for k, v in enumerate(cls)}

    cls2 = capsule(x_batch)[0]
    cls2 = torch.softmax(cls2, dim=-1).tolist()
    labels2 = {k: float(v) for k, v in enumerate(cls2)}
    
    return img, "class {}".format(y), labels, labels2

with gr.Blocks() as demo:

    gr.Markdown("## Image Examples")
    with gr.Row():
        with gr.Column():
            im = gr.Image().style(width=200, height=200)
            capsule_explain = gr.Image(label="Capsule Activation map").style(width=300, height=300)
            conv_explain = gr.Image(label= "Activation map").style(width=300, height=300)
        with gr.Column():
            cls_box = gr.Textbox(label="True Image class")
            label_conv = gr.Label(label="Convolutional Model", num_top_classes=4)
            label_capsule = gr.Label(label="Capsule model", num_top_classes=4)
            btn = gr.Button(value="Load random image")
    btn.click(load_data, inputs=None, outputs=[im, cls_box, label_conv, label_capsule])

   

if __name__ == "__main__":
    demo.launch()