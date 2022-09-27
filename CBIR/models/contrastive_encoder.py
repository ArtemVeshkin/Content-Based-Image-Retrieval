import torch
from torch import nn
from torchsummary import summary
import numpy as np
from typing import List, TypeVar

Tensor = TypeVar('Tensor')


class ContrastiveExtractor(nn.Module):
    def __init__(self,
                 input_size: int = 224,
                 conv_hidden_dims: List = None,
                 conv_out_size: int = 2,
                 fc_hidden_dims: List = None):
        super(ContrastiveExtractor, self).__init__()

        self.input_size = input_size

        if conv_hidden_dims is None:
            conv_hidden_dims = [16, 32, 64, 128, 256]
        modules = []
        in_channels = 3
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
        self.Encoder = nn.Sequential(*modules)

        if fc_hidden_dims is None:
            fc_hidden_dims = [256, 64]
        modules = []
        in_features = 2 * conv_hidden_dims[-1] * conv_out_size ** 2
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

    def forward(self, input1: Tensor, input2: Tensor):
        encoder_out1 = self.Encoder(input1)
        encoder_out2 = self.Encoder(input2)

        fc_stack_input = torch.cat((encoder_out1, encoder_out2), 1)
        return self.fc_stack(fc_stack_input)

    def summary(self):
        summary(self.Encoder, (3, self.input_size, self.input_size))

    def save(self, path, optimizer):
        torch.save({
            'Encoder': self.Encoder.state_dict(),
            'fc_stack': self.fc_stack.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.Encoder.load_state_dict(checkpoint['Encoder'])
        self.fc_stack.load_state_dict(checkpoint['fc_stack'])
        return {'optimizer': checkpoint['optimizer']}
