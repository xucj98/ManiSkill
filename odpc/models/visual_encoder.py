import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18

class Resnet18Encoder(nn.Module):
    def __init__(self, in_channels, out_dim, depth_clamp=0):
        super().__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)
        self.depth_clamp = depth_clamp

    def forward(self, obs):
        """
        obs: dict
            rgb: (B, T, 3*k, H, W)
            depth: (B, T, 1*k, H, W)
        """

        rgb = obs["rgb"].float() / 255.0
        depth = obs["depth"].float()
        if self.depth_clamp > 0:
            depth[depth < 0] = 0
            depth[depth > self.depth_clamp] = 0
        depth = depth / 1024.0
  
        x = torch.cat([rgb, depth], dim=2)  # (B, T, 4*k, H, W)
        batch_size = x.shape[0]
        x = x.flatten(end_dim=1)  # (B*T, 4*k, H, W)
        visual_features = self.resnet(x)  # (B*T, out_dim)
        visual_features = visual_features.view(batch_size, -1)  # (B, T * out_dim)

        return visual_features
        