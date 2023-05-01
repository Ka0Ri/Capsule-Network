import gradio as gr
import cv2
import numpy as np
import yaml
import torch
from main import CapsuleModel
from ReadDataset import affNistread
import random
from pytorch_lightning import LightningModule, Trainer


with open("Capsules/config.yaml", 'r') as stream:
    try:
        PARAMS = yaml.safe_load(stream)
       
    except yaml.YAMLError as exc:
        print(exc)

convmodel = CapsuleModel.load_from_checkpoint(PARAMS["demo-conv"]["training_settings"]["ckpt_path"], PARAMS=PARAMS["demo-conv"])
convmodel.eval()
capsule = CapsuleModel.load_from_checkpoint(PARAMS["demo-cap1"]["training_settings"]["ckpt_path"], PARAMS=PARAMS["demo-cap1"])
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

Test_data = affNistread(mode="test", data_path="data/affMnist", transform=Test_transform, aff=True)

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
   
    weight_softmax = weight_softmax.sum(axis=(2, 3))
    feature_conv = feature_conv[0]
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = normalize(cam)
       
        output_cam.append(cv2.resize(cam, size_upsample))

    img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
    height, width, _ = img.shape
    CAM = cv2.resize(output_cam[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    
    return normalize(result)

def CapsuleCAM(img, feature_conv, weight_softmax, class_idx):

    size_upsample = (40, 40)
    bz, nc, p, h, w = feature_conv.shape
    output_cam = []
    
    weight_softmax = weight_softmax.transpose(1, 0, 2, 3).reshape(-1, nc * p)
    feature_conv = feature_conv[0]
    
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc * p, h * w)))
        cam = cam.reshape(h, w)
        cam = normalize(cam)
       
        output_cam.append(cv2.resize(cam, size_upsample))

    img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
    height, width, _ = img.shape
    CAM = cv2.resize(output_cam[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    
    return normalize(result)

def CapsuleAttentionMap(img, feature_conv, global_feature, class_idx):

    size_upsample = (40, 40)
    bz, h, w, nc, p = feature_conv.shape
    output_cam = []
    
    global_feature = global_feature[0].squeeze()
    feature_conv = feature_conv[0].squeeze()
    # feature_conv = feature_conv
    
    for idx in class_idx:
        cam = feature_conv.dot(global_feature[idx])
        cam = cam.sum(axis=-1)
        cam = cam.reshape(h, w)
        cam = normalize(cam)
       
        output_cam.append(cv2.resize(cam, size_upsample))

    img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
    height, width, _ = img.shape
    CAM = cv2.resize(output_cam[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    
    return normalize(result)
    

def run():
    import time
    for i in range(0, 10):
        time.sleep(1)
        yield load_data()


def load_data():
    index = random.randint(0, 1000)
    x, y = Test_data[index]
    img = transforms.ToPILImage()(x)
    dummy = torch.zeros_like(x)
    x_batch = torch.stack([x, dummy])

    # Convolution
    cls = convmodel(x_batch)[0]
    cls = torch.softmax(cls, dim=-1).tolist()
    labels = {k: float(v) for k, v in enumerate(cls)}

    #visualize CAM
    params = list(convmodel.parameters())
    weight_softmax = np.squeeze(params[-4].tolist())
    features_map = np.array(convmodel.features_blobs[-1])
   
    activation_map = returnCAM(img, features_map, weight_softmax, class_idx=[np.argmax(cls)])
    
    # Capsule
    cls2 = capsule(x_batch)[0]
    cls2 = torch.softmax(cls2, dim=-1).tolist()
    labels2 = {k: float(v) for k, v in enumerate(cls2)}

    features_map = np.array(capsule.features_blobs[-2])
    global_map = np.array(capsule.features_blobs[-1])

    activation_map2 = CapsuleAttentionMap(img, features_map, global_map, class_idx=[np.argmax(cls2)])

    return img, "class {}".format(y), labels, labels2, activation_map, activation_map2

with gr.Blocks() as demo:

    gr.Markdown("## Image Examples")
    with gr.Row():
        with gr.Column():
            im = gr.Image().style(width=200, height=200)
            conv_explain = gr.Image(label= "Activation map").style(width=300, height=300)
            capsule_explain = gr.Image(label="Capsule Activation map").style(width=300, height=300)
        with gr.Column():
            cls_box = gr.Textbox(label="True Image class")
            label_conv = gr.Label(label="Convolutional Model", num_top_classes=4)
            label_capsule = gr.Label(label="Capsule model", num_top_classes=4)
            btn = gr.Button(value="Load random image")
    btn.click(run, inputs=None, outputs=[im, cls_box, label_conv, label_capsule, conv_explain, capsule_explain])



if __name__ == "__main__":
    # load_data()
    demo.queue()
    demo.launch(share=True)