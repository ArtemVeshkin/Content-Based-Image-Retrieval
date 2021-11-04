import torch
from torchsummary import summary
from VAE import VAE


def fit_VAE(cfg):
    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    input_tensor = torch.FloatTensor(10, 3, 224, 224)
    vae.forward(input_tensor)
