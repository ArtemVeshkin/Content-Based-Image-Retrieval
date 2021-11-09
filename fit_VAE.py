import torch
from models import VAE
from models import BatchGenerator
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np


def fit_VAE(cfg):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    vae.move_to_device(device)

    vae.summary()
    print_train_info(cfg)

    batch_generator = BatchGenerator(cfg.image_dir, batch_size=cfg.batch_size)
    params = vae.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr)

    if cfg.load_checkpoint:
        load_dict = vae.load(to_absolute_path(cfg.load_path))
        optimizer.load_state_dict(load_dict['optimizer'])
        print(f"Loaded checkpoint from {to_absolute_path(cfg.load_path)}\n")

    for i in range(cfg.max_steps):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        batch = batch.to(device)
        loss = vae.loss_function(*vae.forward(batch), **{'M_N': cfg.KLD_weigth})
        print(f"Step {i}: loss = {loss['loss']:0.4f} | "
              f"Reconstruction part = {loss['Reconstruction_Loss']:0.4f} | "
              f"KLD part = {loss['KLD']:0.4f}")

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        if i % cfg.save_every_steps == 0 and i > 0:
            vae.save(path=to_absolute_path(cfg.save_path), optimizer=optimizer)
    # sample = vae.sample(1, device).detach().numpy()[0]
    # sample = 0.3 * sample[0, :, :] + 0.59 * sample[1, :, :] + 0.11 * sample[2, :, :]
    # plt.imshow(sample)
    # plt.show()


def print_train_info(cfg):
    print("\n=====TRAINING STARTED=====")
    if torch.cuda.is_available():
        print(f"Running on {torch.cuda.get_device_name(0)}")
    else:
        print(f"Running on CPU")
    print(f"Batch size = {cfg.batch_size}")
    print(f"Learning rate = {cfg.lr}")
    print(f"Max train steps = {cfg.max_steps}")
    print()
