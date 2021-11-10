import torch
from models import VAE
from models import BatchGenerator
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from PIL import Image
from kernel.utils import normalize_image


def fit_VAE(cfg):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    vae.move_to_device(device)

    # vae.summary()
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
        sample_from_batch = vae.sample_from_image(batch)
        mse = F.mse_loss(batch, sample_from_batch).item()
        print(f"Step {i}: loss = {loss['loss']:0.6f} | "
              f"Reconstruction part = {loss['Reconstruction_Loss']:0.6f} | "
              f"KLD part = {loss['KLD']:0.6f} | "
              f"MSE = {mse:0.6f}")

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        if i % cfg.save_every_steps == 0 and i > 0:
            vae.save(path=to_absolute_path(cfg.save_path), optimizer=optimizer)

    if cfg.sample_after_training:
        f, axarr = plt.subplots(2, 1)
        image = batch_generator.get_batch()
        image = torch.FloatTensor(image)
        sample = vae.sample_from_image(image)
        sample = sample[0].detach().numpy()
        image = image[0].detach().numpy()
        image = np.moveaxis(image, 0, -1)
        sample = np.moveaxis(sample, 0, -1)
        axarr[0].imshow(normalize_image(image * 255))
        axarr[1].imshow(normalize_image(sample * 255))
        plt.show()


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
