import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
from torchsummary import summary
from skimage.metrics import structural_similarity as ssim_s
from .metrics import ssim
import numpy as np

Tensor = TypeVar('Tensor')


class AutoEncoder:
    def __init__(self,
                 input_size: int,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 n_conv_layers: int = 1,
                 **kwargs) -> None:
        self.latent_dim = latent_dim
        self.input_size = input_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder_last_hidden_dim = hidden_dims[-1]

        # Build Encoder
        for h_dim in hidden_dims:
            conv_seq = [nn.Conv2d(h_dim, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU()]
            additional_layers = []
            for _ in range(n_conv_layers - 1):
                for elem in conv_seq:
                    additional_layers.append(elem)
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    *additional_layers
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_out_size = int(input_size / 2 ** len(hidden_dims))
        self.fc_hidden = nn.Linear(hidden_dims[-1] * self.encoder_out_size ** 2, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.encoder_out_size ** 2)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            conv_seq = [nn.Conv2d(h_dim, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU()]
            additional_layers = []
            for _ in range(n_conv_layers - 1):
                for elem in conv_seq:
                    additional_layers.append(elem)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[i + 1], out_channels=hidden_dims[i + 1],
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def save(self, path, optimizer):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'fc_hidden': self.fc_hidden.state_dict(),
            'decoder_input': self.decoder_input.state_dict(),
            'final_layer': self.final_layer.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.fc_hidden.load_state_dict(checkpoint['fc_hidden'])
        self.decoder_input.load_state_dict(checkpoint['decoder_input'])
        self.final_layer.load_state_dict(checkpoint['final_layer'])

        return {'optimizer': checkpoint['optimizer']}

    def move_to_device(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_hidden.to(device)
        self.decoder_input.to(device)
        self.final_layer.to(device)

    def get_params(self):
        params = [*self.encoder.parameters(),
                  *self.decoder.parameters(),
                  *self.fc_hidden.parameters(),
                  *self.decoder_input.parameters(),
                  *self.final_layer.parameters()]
        return params

    def summary(self):
        print(f"=====MODEL SUMMARY=====")
        tensor = torch.FloatTensor(10, 3, self.input_size, self.input_size)
        if torch.cuda.is_available():
            tensor = tensor.to(torch.device("cuda"))
        print("Encoder:")
        summary(self.encoder, tensor.shape[1:])
        tensor = self.encode(tensor)
        tensor = self.decoder_input(tensor)
        tensor = tensor.view(-1, self.encoder_last_hidden_dim,
                             self.encoder_out_size, self.encoder_out_size)
        print("\nDecoder:")
        summary(self.decoder, tensor.shape[1:])
        tensor = self.decoder(tensor)
        print("\nFinal layer:")
        summary(self.final_layer, tensor.shape[1:])

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc_hidden(result)

        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_last_hidden_dim,
                             self.encoder_out_size, self.encoder_out_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoded = self.encode(input)
        return [self.decode(encoded), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        recons_loss_type = kwargs['recons_loss']
        if recons_loss_type == 'mse':
            recons_loss = F.mse_loss(recons, input)
        elif recons_loss_type == 'mae':
            recons_loss = torch.mean(torch.abs(recons - input))
        elif recons_loss_type == 'mae_max':
            recons_loss = torch.max(torch.mean(torch.abs(recons - input), dim=(1, 2, 3)))
        elif recons_loss_type == 'ssim':
            recons_loss = -ssim(input, recons)
        elif recons_loss_type == 'ce':
            recons_loss = -torch.mean(input * torch.log(recons) + (1 - input) * torch.log(1 - recons))
        else:
            raise ValueError(f"Unknown recons loss type: {recons_loss_type}")
        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': 0, 'var': 0}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def sample_from_image(self, image):
        encoded = self.encoder(image)
        decoded = self.decoder(encoded)
        return self.final_layer(decoded)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
