import torch
from torchsummary import summary
from models.VAE import VAE


def fit_VAE(cfg):
    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    input_tensor = torch.FloatTensor(10, 3, 224, 224)
    params = vae.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr)
    for i in range(cfg.max_steps):
        loss = vae.loss_function(*vae.forward(input_tensor), **{'M_N': 1})
        print(f"Step {i}: loss = {loss['loss']}")
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
