import torch
import torch.nn as nn

from einops import rearrange

from odpc.utils import utils
from odpc.models.vision.base_encoder import BaseVisionEncoder


class PointNetEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int=3,
            out_channels: int=1024,
            use_layernorm: bool=True,
            final_norm: str='layernorm',
            use_projection: bool=True,
    ):
        """
        Args:
            in_channels (int): feature size of input (3 or 6)
        """
        super().__init__()
        block_channel = [64, 128, 256]
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(BaseVisionEncoder):
    def __init__(
            self,
            obs_meta: dict,
            use_layernorm: bool=True,
            final_norm: str='layernorm',
            use_projection: bool=True,
    ):
        self.use_layernorm = use_layernorm
        self.final_norm = final_norm
        self.use_projection = use_projection
        super().__init__(obs_meta)
        
    def _init_encoders(self):
        encoders = {}
        for name, meta in self.obs_meta.items():
            if meta["type"] == "point_cloud":
                ic = meta["shape"][-1]
                oc = meta["output_dim"]
                encoder = PointNetEncoder(
                    in_channels=ic, 
                    out_channels=oc,
                    use_layernorm=self.use_layernorm,
                    final_norm=self.final_norm,
                    use_projection=self.use_projection,
                )
            encoders[name] = encoder
        return nn.ModuleDict(encoders)
    
    def forward(self, obs):
        features = []
        for name, meta in self.obs_meta.items():
            data = utils.get_nested_value(obs, meta["point_cloud"])
            data = data[..., :meta["shape"][-1]]
            if data.ndim == 4:
                b, t, n, c = data.shape
                data = rearrange(data, "b t n c -> (b t) n c")
                feature = self.encoders[name](data)
                feature = rearrange(feature, "(b t) d -> b (t d)", b=b)
            else:
                feature = self.encoders[name](data)
            features.append(feature)
        features = torch.cat(features, dim=-1)
        return features

        