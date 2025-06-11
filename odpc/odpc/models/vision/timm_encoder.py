import torch
from torch import nn
import timm
import torchvision.transforms as T
from einops import rearrange

from .base_encoder import BaseVisionEncoder
from ...utils import utils


class TimmEncoder(BaseVisionEncoder):
    def __init__(
            self,
            obs_meta: dict,
            model_name: str = "vit_large_patch14_dinov2.lvd142m",
            pretrained: bool = True,
            img_size: int = 224,  # DINOv2默认输入尺寸
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.img_size = img_size
        super().__init__(obs_meta)

    def _init_encoders(self):
        encoders = {}
        for name, meta in self.obs_meta.items():
            if meta["type"] in ["rgb", "rgbd"]:
                ic = meta["shape"][-3]
                oc = meta["output_dim"]
                
                # 创建timm模型
                encoder = timm.create_model(
                    self.model_name,
                    pretrained=self.pretrained,
                    in_chans=ic,
                    num_classes=oc,
                    img_size=self.img_size
                )
                
                # 如果模型是Vision Transformer类型，需要确保输入通道数正确
                if hasattr(encoder, 'patch_embed'):
                    encoder.patch_embed.proj = nn.Conv2d(
                        ic, 
                        encoder.patch_embed.proj.out_channels,
                        kernel_size=encoder.patch_embed.proj.kernel_size,
                        stride=encoder.patch_embed.proj.stride,
                        padding=encoder.patch_embed.proj.padding,
                        bias=encoder.patch_embed.proj.bias is not None
                    )
                
                encoders[name] = encoder
        return nn.ModuleDict(encoders)

    def forward(self, obs):
        features = []
        for name, meta in self.obs_meta.items():
            data = None
            if meta["type"] == "rgbd":
                if "rgbd" in meta:
                    data = utils.get_nested_value(obs, meta["rgbd"])
                elif "rgb" in meta and "depth" in meta:
                    rgb = utils.get_nested_value(obs, meta["rgb"])
                    depth = utils.get_nested_value(obs, meta["depth"])
                    data = torch.cat([rgb, depth], dim=-3)
                else:
                    raise RuntimeError(f"unable parse metadata")
            elif meta["type"] == "rgb":
                data = utils.get_nested_value(obs, meta["rgb"])
            
            if data.ndim == 5:
                b, t, c, h, w = data.shape
                data = rearrange(data, "b t c h w -> (b t) c h w")
                
                # 调整图像尺寸
                if h != self.img_size or w != self.img_size:
                    data = T.functional.resize(data, [self.img_size, self.img_size])
                
                feature = self.encoders[name](data)
                feature = rearrange(feature, "(b t) d -> b (t d)", b=b)
            else:
                # 调整图像尺寸
                if data.shape[-2] != self.img_size or data.shape[-1] != self.img_size:
                    data = T.functional.resize(data, [self.img_size, self.img_size])
                feature = self.encoders[name](data)
            
            features.append(feature)
        features = torch.cat(features, dim=-1)
        return features
