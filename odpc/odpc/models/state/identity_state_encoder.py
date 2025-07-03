from torch import nn

from odpc.models.state.base_state_encoder import BaseStateEncoder

class IdentityStateEncoder(BaseStateEncoder):
    def __init__(self, obs_meta: dict):
        super().__init__(obs_meta)

    def _init_encoders(self):
        encoders = {}
        for name, meta in self.obs_meta.items():
            encoders[name] = nn.Identity()
        return encoders
