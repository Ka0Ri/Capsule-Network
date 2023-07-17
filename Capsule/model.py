import torch
import torch.nn as nn
from .CapsuleLayer import AdaptiveCapsuleHead

# MODEL For Classification
from torchvision.models import resnet18, resnet50, resnet152, \
                                densenet121, densenet161, densenet201, \
                                efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, \
                                inception_v3, \
                                swin_t, swin_b, swin_s
                                
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, \
                                DenseNet121_Weights, DenseNet161_Weights, DenseNet201_Weights, \
                                EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, \
                                Inception_V3_Weights, \
                                Swin_S_Weights, Swin_T_Weights, Swin_B_Weights


# MODEL For Segmentation
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
                                            deeplabv3_resnet50, deeplabv3_resnet101

from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights, \
                                            DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights \


MODEL = {
        "resnet-s": resnet18, "resnet-m": resnet50, "resnet-l": resnet152,
        "densenet-s": densenet121, "densenet-m": densenet161, "densenet-l": densenet201,
        "efficientnet-s": efficientnet_v2_s, "efficientnet-m": efficientnet_v2_m, "efficientnet-l": efficientnet_v2_l,
        "inception": inception_v3,
        "swin-s": swin_t, "swin-m": swin_s, "swin-l": swin_b,

        "fcn-m": fcn_resnet50, "fcn-l": fcn_resnet101,
        "deeplab-m": deeplabv3_resnet50, "deeplab-l": deeplabv3_resnet101,
        }

WEIGHTS = {
            "resnet-s": ResNet18_Weights, "resnet-m": ResNet50_Weights, "resnet-l": ResNet152_Weights,
           "densenet-s": DenseNet121_Weights, "densenet-m": DenseNet161_Weights, "densenet-l": DenseNet201_Weights,
           "efficientnet-s": EfficientNet_V2_S_Weights, "efficientnet-m": EfficientNet_V2_M_Weights, "efficientnet-l": EfficientNet_V2_L_Weights,
           "inception": Inception_V3_Weights,
           "swin-s": Swin_T_Weights, "swin-m": Swin_S_Weights, "swin-l": Swin_B_Weights,

           "fcn-m": FCN_ResNet50_Weights, "fcn-l": FCN_ResNet101_Weights,
            "deeplab-m": DeepLabV3_ResNet50_Weights, "deeplab-l": DeepLabV3_ResNet101_Weights
        }


class Simple_Classifier(nn.Module):
    '''
    Simple Classifier for Classification
    
    Args:
        num_ftrs: number of features
        n_cls: number of classes
        n_emb: number of embedding
        n_layers: number of layers
    '''
    def __init__(self, num_ftrs, n_cls, n_emb=512, n_layers=1) -> None:
        super(Simple_Classifier, self).__init__()
        if(n_layers == 1):
            self.classifier =  nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Conv2d(num_ftrs, n_cls, 1))
        else:
            self.classifier =  nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Conv2d(num_ftrs, n_emb, 1),
                                            nn.ReLU())
            for i in range(1, n_layers):
                if(i == n_layers - 1):
                    self.classifier.append(nn.Conv2d(n_emb, n_cls, 1))
                else:
                    self.classifier.append(nn.Conv2d(n_emb, n_emb, 1))
                    self.classifier.append(nn.ReLU())

    def forward(self, x):
        return self.classifier(x)



class CapsuleWrappingClassifier(nn.Module):
    '''
    Wrapping Classification model with Capsule head,
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 
    in Capsule mode (is_caps = True) or in standard mode
    - Supported models: resnet, shuffle, inception, resnext, vgg, mobile, , 
                        vit, swin, wideresnet, dense, convnext, efficient
    [-option s: small, m: medium, l: large]
    Authors: dtvu1707@gmail.com
    '''
    def __init__(self, model_configs=None):

        super(CapsuleWrappingClassifier, self).__init__()
        # Parse parameters
        self.is_caps = model_configs['head']['is_caps']
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.is_backbone = model_configs['backbone']['is_backbone'] # TODO: support for feats (temoporal)
        self.n_cls = model_configs['head']['n_cls']
        self.n_layers = model_configs['head']['n_layers']
        self.n_emb = model_configs['head']['n_emb']

        # Load backbone
        self.backbone = self._model_selection()
        if not self.is_backbone:
            self.backbone = nn.Identity()
        if not self.is_full:
            if self.is_caps:
                self.classifier =  nn.Sequential(
                                            AdaptiveCapsuleHead(self.num_ftrs, 
                                                               head=model_configs['head']),
                                            nn.Flatten(start_dim=1)
                                        )
            else:
                self.classifier =  nn.Sequential(
                                            Simple_Classifier(self.num_ftrs, self.n_cls,
                                                        self.n_emb, self.n_layers),
                                            nn.Flatten(start_dim=1))
               
    def _model_selection(self):

        # Load pretrained model
        name = self.backbone_name
        assert name in MODEL.keys(), "Model %s not found" % name

        if self.is_pretrained:
            base_model = MODEL[name](weights=WEIGHTS[name].IMAGENET1K_V1)
        else:
            base_model = MODEL[name](weights=None)
        self.preprocess = WEIGHTS[name].IMAGENET1K_V1.transforms()
        self.meta = WEIGHTS[name].IMAGENET1K_V1.meta

        # turn off gradient
        if self.is_freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        if not self.is_full:
            # Remove Classification head
            if any([x in name for x in ["resnet", "inception"]]):
                if "inception" in name: 
                    base_model.aux_logits=False
                self.num_ftrs = base_model.fc.in_features
                base_model.fc = nn.Identity()
            elif any([x in name for x in ["densenet"]]):
                self.num_ftrs = base_model.classifier.in_features
                base_model = base_model.features
            elif "swin" in name:
                self.num_ftrs = base_model.head.in_features
                base_model.head = nn.Identity()

        return base_model

    def forward(self, x, y=None):

        feats = self.backbone(x)
        if self.is_full:
            return feats
        else:
            if(len(feats.shape) < 4):
                s = int((feats.shape[1] // self.num_ftrs) ** 0.5)
                feats = feats.reshape(-1, self.num_ftrs, s, s)
            return self.classifier(feats)

class CapsuleWrappingSegment(nn.Module):
    '''
    Wrapping Segmentation model with Capsule head, 
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 
    in Capsule mode (is_caps = True) or in standard mode

    - Supported models: fcn, deeplab [-option m: medium, l: large]
    Authors: dtvu1707@gmail.com
    '''
    def __init__(self, model_configs: dict) -> None:

        super(CapsuleWrappingSegment, self).__init__()
        # Parse parameters
        self.is_caps = model_configs['head']['is_caps']
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.is_backbone = model_configs['backbone']['is_backbone'] # TODO: support for feats (temoporal)
        self.n_cls = model_configs['head']['n_cls']
        self.n_layers = model_configs['head']['n_layers']
        self.n_emb = model_configs['head']['n_emb']

        # Load backbone
        self.backbone = self._model_selection()
        if not self.is_full:
            if self.is_caps:
                self.backbone.classifier = AdaptiveCapsuleHead(self.num_ftrs, 
                                                                head=model_configs['head'])
            else:
                self.backbone.classifier = Simple_Classifier(self.num_ftrs, self.n_cls,
                                                            self.n_emb, self.n_layers)
        
    def _model_selection(self):

        # Load pretrained model
        name = self.backbone_name
        assert name in MODEL, "Model %s not found" % name

        if self.is_pretrained:
            base_model = MODEL[name](weights=WEIGHTS[name].COCO_WITH_VOC_LABELS_V1)
        else:
            base_model = MODEL[name](weights=None)
        self.preprocess = WEIGHTS[name].COCO_WITH_VOC_LABELS_V1.transforms()
        self.meta = WEIGHTS[name].COCO_WITH_VOC_LABELS_V1.meta
        
        # turn off gradient
        if self.is_freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        if not self.is_full:
            # Modify last layer
            if any([x in name for x in ["fcn", "deeplab"]]):
                base_model.aux_classifier = None
                if "fcn" in name: 
                    self.num_ftrs = base_model.classifier[0].in_channels
                else: 
                    self.num_ftrs = base_model.classifier[0].convs[0][0].in_channels                    
                base_model.classifier = nn.Identity()

        return base_model
       
    def forward(self, x, y=None):

        a = self.backbone(x)['out']
        return a
    
