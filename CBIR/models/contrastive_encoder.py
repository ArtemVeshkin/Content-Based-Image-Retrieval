import torch
from torch import nn
from torchsummary import summary
import numpy as np
from typing import List, TypeVar
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms import Resize, InterpolationMode

Tensor = TypeVar('Tensor')


class ContrastiveExtractor(nn.Module):
    def __init__(self,
                 input_size: int = 112,
                 conv_hidden_dims: List = None,
                 conv_out_size: int = 2,
                 fc_hidden_dims: List = None,
                 embedding_size: int = 128):
        super(ContrastiveExtractor, self).__init__()

        self.input_size = input_size

        # if conv_hidden_dims is None:
        #     conv_hidden_dims = [8, 16, 32, 64, 128]
        # modules = []
        # in_channels = 3
        # for h_dim in conv_hidden_dims:
        #     modules.append(nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels=h_dim,
        #                   kernel_size=3, stride=2, padding=1),
        #         nn.BatchNorm2d(h_dim),
        #         nn.LeakyReLU(),
        #     ))
        #     in_channels = h_dim
        #
        # modules.append(nn.Sequential(
        #     nn.AdaptiveAvgPool2d(conv_out_size),
        #     nn.Flatten()
        # ))
        #
        # if fc_hidden_dims is None:
        #     fc_hidden_dims = [128]
        # in_features = conv_hidden_dims[-1] * conv_out_size ** 2
        # for h_dim in fc_hidden_dims:
        #     modules.append(nn.Sequential(
        #         nn.Linear(in_features, h_dim),
        #         nn.LeakyReLU()
        #     ))
        #     in_features = h_dim
        # modules.append(nn.Linear(in_features, embedding_size))

        # self.Encoder = nn.Sequential(*modules)
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.Encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,

            model.avgpool,
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=embedding_size, bias=True)
        )

    def forward(self, input1: Tensor, input2: Tensor):
        resizer = Resize(size=self.input_size, interpolation=InterpolationMode.BICUBIC)
        if input1.shape[-1] != self.input_size:
            input1 = resizer(input1)
        if input2.shape[-1] != self.input_size:
            input2 = resizer(input2)

        encoder_out1 = self.Encoder(input1)
        encoder_out2 = self.Encoder(input2)
        return encoder_out1, encoder_out2

    def summary(self):
        summary(self.Encoder, (3, self.input_size, self.input_size))

    def save(self, path, optimizer):
        torch.save({
            'Encoder': self.Encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.Encoder.load_state_dict(checkpoint['Encoder'])
        return {'optimizer': checkpoint['optimizer']}
