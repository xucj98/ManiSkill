import torch
from torch import nn

from torchvision.models import resnet18

from .base_encoder import BaseVisionEncoder



class Resnet18Encoder(BaseVisionEncoder):
    def __init__(
            self,
            obs_meta: dict,
    ):
        super().__init__(obs_meta)

    def _init_encoders(self):
        encoders = {}
        for name, meta in self.obs_meta.items():
            if meta["type"] in ["rgb", "rgbd", "stack_image"]:
                ic = meta["shape"][-3]
                oc = meta["output_dim"]
                encoder = resnet18()
                encoder.conv1 = nn.Conv2d(
                    ic, 64, kernel_size=7, stride=2, padding=3, bias=False)
                encoder.fc = nn.Linear(encoder.fc.in_features, oc)
                encoders[name] = encoder
        return nn.ModuleDict(encoders)

        