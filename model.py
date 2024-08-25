"""
ML AI for the connect 4
"""

import torch
from torch import nn

from arguments import Arguments


class Model(nn.Module):
    """
    Model for the hnefatafl AI
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.in_dims = args.in_dims
        self.out_dims = args.out_dims
        self.dtype = args.dtype

        self.conv = torch.nn.Sequential(
            nn.ZeroPad2d(1),  # 6 x 7 x 3
            nn.Conv2d(self.in_dims, self.in_dims * 4, 5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(2),  # 3 x 3
            nn.Dropout(0.1),
            nn.Conv2d(self.in_dims * 4, self.in_dims * 8, 5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(2),  # 1 , 1
            nn.Dropout(0.05),
        )

        self.ff = torch.nn.Sequential(
            nn.LazyLinear(256),
            nn.SELU(),
            nn.Dropout(0.05),
        )

        self.value = torch.nn.Sequential(
            nn.LazyLinear(128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self.policy = torch.nn.Sequential(
            nn.LazyLinear(128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.out_dims),
        )

    def forward(self, x: torch.Tensor):
        """
        Measure policy and value network
        """
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.ff(x)

        value = self.value(x)
        policy = self.policy(x)

        return policy, value

    def predict(self, board: torch.Tensor):
        device = torch.device("mps")
        x = board.type(torch.float32).to(device)
        x = x.permute(2, 0, 1)
        x = x.view(1, *x.shape)
        self.eval()  # Disable training mode
        with torch.no_grad():
            policy, value = self.forward(x)

        return policy.cpu().numpy()[0, :], value.cpu().numpy()[0, :]
