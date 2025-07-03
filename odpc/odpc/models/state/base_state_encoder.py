import torch
from torch import nn

from einops import rearrange

from odpc.utils import utils

class BaseStateEncoder(nn.Module):
    def __init__(
            self, 
            obs_meta: dict,
    ):
        super().__init__()
        self.obs_meta = obs_meta
        self.encoders: nn.ModuleDict = self._init_encoders()

    def _init_encoders(self):
        return nn.ModuleDict()

    def forward(self, obs):
        features = []
        for name, meta in self.obs_meta.items():
            data = None
            if meta["type"] == "stack_state":
                states = []
                for key in meta["states"]:
                    state = utils.get_nested_value(obs, key)
                    states.append(state)
                data = torch.cat(states, dim=-1)

            if data.ndim == 3:
                b, t, d = data.shape
                data = rearrange(data, "b t d -> (b t) d")
                feature = self.encoders[name](data)
                feature = rearrange(feature, "(b t) d -> b (t d)", b=b)
            else:
                feature = self.encoders[name](data)

            features.append(feature)
        features = torch.cat(features, dim=-1)
        return features
