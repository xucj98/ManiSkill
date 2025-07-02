import torch
from torch import nn
from einops import rearrange
from torchvision.transforms.functional import resize

from odpc.utils import utils


class BaseVisionEncoder(nn.Module):
    def __init__(
            self, 
            obs_meta: dict,
    ) -> None:
        super().__init__()
        self.obs_meta = obs_meta
        self.encoders: nn.ModuleDict = self._init_encoders()

    def _init_encoders(self):
        return nn.ModuleDict()

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
            elif meta["type"] == "stack_image":
                channles = []
                for c_key in meta["channels"]:
                    channles.append(utils.get_nested_value(obs, c_key))
                data = torch.cat(channles, dim=-3)
            
            if data.ndim == 5:
                b, t, c, h, w = data.shape

                data = rearrange(data, "b t c h w -> (b t) c h w")
                
                # 调整图像尺寸
                _, cd, hd, wd = meta.shape
                assert c == cd, "channel number mismatch"
                if h != hd or w != wd:
                    data = resize(data, [hd, wd])
                
                feature = self.encoders[name](data)
                feature = rearrange(feature, "(b t) d -> b (t d)", b=b)
            else:
                # 调整图像尺寸
                _, cd, hd, wd = meta.shape
                assert c == cd, "channel number mismatch"
                if h != hd or w != wd:
                    data = resize(data, [hd, wd])

                feature = self.encoders[name](data)
            
            features.append(feature)
        features = torch.cat(features, dim=-1)
        return features




