import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.Module):
    def __init__(self, in_dims: tuple[int, int, int], out_dims: int, device="cpu"):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        self.conv = torch.nn.Sequential(
            nn.ZeroPad2d(1), # 12 x 12 x 3
            nn.Conv2d(in_dims[2], in_dims[2] * 2, 3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2), # 6 x 6
            nn.Conv2d(in_dims[2] * 2, in_dims[2] * 4, 3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2), # 3 x 3
        ).to(device)

        self.ff = torch.nn.Sequential(
            nn.LazyLinear(128),
            nn.SELU(),
        ).to(device)

        self.value = torch.nn.Sequential(
            nn.LazyLinear(128),
            nn.SELU(),
            nn.LazyLinear(1),
            nn.Tanh(),
        ).to(device)

        self.policy = torch.nn.Sequential(
            nn.LazyLinear(128),
            nn.SELU(),
            nn.LazyLinear(out_dims),
            nn.Softmax(dim=1),
        ).to(device)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.ff(x)

        value = self.value(x)
        policy = self.policy(x)

        return policy, value
