import torch
from torchsummary import summary
from models import VAE
from models import BatchGenerator


def fit_VAE(cfg):
    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)

    batch_generator = BatchGenerator(cfg.image_dir, batch_size=20)
    params = vae.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr)
    for i in range(cfg.max_steps):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        loss = vae.loss_function(*vae.forward(batch), **{'M_N': 1})
        print(f"Step {i}: loss = {loss['loss']}")

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

