from torch import nn

from odpc.models.state.base_state_encoder import BaseStateEncoder

class MLPStateEncoder(BaseStateEncoder):
    def __init__(self, obs_meta: dict):
        super().__init__(obs_meta)

    def _init_encoders(self):
        encoders = {}
        for name, meta in self.obs_meta.items():
            in_dim = meta["shape"][-1]
            out_dim = meta["output_dim"]
            layers = []
            for dim in meta["hidden_dims"]:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(nn.ReLU())
                in_dim = dim
            layers.append(nn.Linear(in_dim, out_dim))
            encoders[name] = nn.Sequential(*layers)
        return encoders
