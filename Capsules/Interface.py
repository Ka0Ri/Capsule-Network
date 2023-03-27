import gradio as gr
import os
import cv2
import numpy as np
import yaml
import torch
from main import CapsuleModel
from ReadDataset import affNistread
from torch.utils.data import DataLoader
import random
from PIL import Image
from pytorch_lightning import LightningModule, Trainer


with open("Capsules/config.yaml", 'r') as stream:
    try:
        PARAMS = yaml.safe_load(stream)
       
    except yaml.YAMLError as exc:
        print(exc)

# model = CapsuleModel(PARAMS)
convmodel = CapsuleModel.load_from_checkpoint("model/affNist-conv/epoch=193-step=15326.ckpt", PARAMS=PARAMS["gradio"])
convmodel.eval()
capsule = CapsuleModel.load_from_checkpoint("model/affNist-fuzzy-ce/epoch=5-step=474.ckpt", PARAMS=PARAMS["gradio1"])
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

def normalize(img):
    img = img - np.min(img)
    nor_img = img / np.max(img)
    nor_img = np.uint8(255 * nor_img)
    return nor_img




def returnCAM(img, feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (40, 40)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
   
    weight_softmax = weight_softmax[class_idx].sum(axis=(2, 3))
    for idx in class_idx:
        cam = weight_softmax.dot(feature_conv[0].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = normalize(cam)
       
        output_cam.append(cv2.resize(cam, size_upsample))

    img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
    height, width, _ = img.shape
    CAM = cv2.resize(output_cam[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    
    return normalize(result)

def predict(img, model, x_batch):

    cls = model(x_batch)[0]
    cls = torch.softmax(cls, dim=-1).tolist()
    labels = {k: float(v) for k, v in enumerate(cls)}

    #visualize CAM
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-4].tolist())
    features_map = np.array(model.features_blobs[-1])
   
    activation_map = returnCAM(img, features_map, weight_softmax, class_idx=[np.argmax(cls)])
    
    return labels, activation_map


def load_data():
    index = random.randint(0, 1000)
    x, y = Test_data[index]
    img = transforms.ToPILImage()(x)
    dummy = torch.zeros_like(x)
    x_batch = torch.stack([x, dummy])

    labels, activation_map = predict(img, convmodel, x_batch)
    
    # labels2, activation_map2 = predict(img, capsule, x_batch)
    cls2 = capsule(x_batch)[0]
    cls2 = torch.softmax(cls2, dim=-1).tolist()
    labels2 = {k: float(v) for k, v in enumerate(cls2)}
    
    return img, "class {}".format(y), labels, labels2, activation_map

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
    btn.click(load_data, inputs=None, outputs=[im, cls_box, label_conv, label_capsule, conv_explain])

   

if __name__ == "__main__":
    # load_data()
    demo.launch()