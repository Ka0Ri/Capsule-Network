import torch.nn as nn
from .CapsuleLayer import AdaptiveCapsuleHead

from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
                                            deeplabv3_resnet50, deeplabv3_resnet101

from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights, \
                                            DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights \
                                            

MODEL = {
    "fcn-m": fcn_resnet50, "fcn-l": fcn_resnet101,
    "deeplab-m": deeplabv3_resnet50, "deeplab-l": deeplabv3_resnet101,
}

WEIGHTS = {
    "fcn-m": FCN_ResNet50_Weights, "fcn-l": FCN_ResNet101_Weights,
    "deeplab-m": DeepLabV3_ResNet50_Weights, "deeplab-l": DeepLabV3_ResNet101_Weights
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

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
        self.is_caps = model_configs["is_caps"]
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.n_cls = model_configs['n_cls']
        
        if self.is_caps:
            self.caps = model_configs['caps']
            self.cap_dim = self.caps['cap_dims']
            self.mode = self.caps['mode']
            self.routing_config = [self.caps['routing']['type']] + self.caps['routing']["params"]
        
        self._model_selection()

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
                if self.is_caps:
                    if "fcn" in name: 
                        num_ftrs = base_model.classifier[0].in_channels
                    else: 
                        num_ftrs = base_model.classifier[0].convs[0][0].in_channels
                    base_model.classifier = AdaptiveCapsuleHead(num_ftrs, self.n_cls, self.cap_dim, 
                                                        self.mode, False, *self.routing_config)
                else:
                    num_ftrs = base_model.classifier[-1].in_channels
                    base_model.classifier[-1] = nn.Conv2d(num_ftrs, self.n_cls, 1)

        self.base_model = base_model
       
    def forward(self, x, y=None):

        a = self.base_model(x)['out']
        return a