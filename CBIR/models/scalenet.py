import torch
from torch import nn
from torchsummary import summary
import numpy as np
from typing import List, TypeVar

Tensor = TypeVar('Tensor')


class ScaleNet(nn.Module):
    def __init__(self,
                 input_size: int = 224,
                 grayscale_input: bool = False,
                 conv_hidden_dims: List = None,
                 conv_out_size: int = 2,
                 fc_hidden_dims: List = None):
        super(ScaleNet, self).__init__()

        self.input_size = input_size
        in_channels = 2 if grayscale_input else 6
        self.in_channels = in_channels

        if conv_hidden_dims is None:
            conv_hidden_dims = [16, 32, 64, 128, 256]
        modules = []
        for h_dim in conv_hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            ))
            in_channels = h_dim

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(conv_out_size),
            nn.Flatten()
        ))
        self.conv_stack = nn.Sequential(*modules)

        if fc_hidden_dims is None:
            fc_hidden_dims = [256, 64]
        modules = []
        in_features = conv_hidden_dims[-1] * conv_out_size ** 2
        for h_dim in fc_hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_features, h_dim),
                nn.LeakyReLU()
            ))
            in_features = h_dim

        modules.append(nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        ))
        self.fc_stack = nn.Sequential(*modules)

    def forward(self, input: Tensor):
        conv_out = self.conv_stack(input)
        return self.fc_stack(conv_out)

    def summary(self):
        summary(self, (self.in_channels, self.input_size, self.input_size))

    def save(self, path, optimizer):
        torch.save({
            'conv_stack': self.conv_stack.state_dict(),
            'fc_stack': self.fc_stack.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.conv_stack.load_state_dict(checkpoint['conv_stack'])
        self.fc_stack.load_state_dict(checkpoint['fc_stack'])
        return {'optimizer': checkpoint['optimizer']}
