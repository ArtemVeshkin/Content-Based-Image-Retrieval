import torch
from models import VAE
from models import BatchGenerator
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from kernel.utils import normalize_image
from skimage.transform import resize


def fit_VAE(cfg):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    vae.move_to_device(device)

    print_train_info(cfg)
    # vae.summary()

    batch_generator = BatchGenerator(cfg.image_dir, batch_size=cfg.batch_size,
                                     skip_background=cfg.skip_background,
                                     use_MNIST=cfg.use_MNIST, input_size=cfg.input_size)
    params = vae.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr, weight_decay=1e-5)

    if cfg.load_checkpoint:
        load_dict = vae.load(to_absolute_path(cfg.load_path))
        optimizer.load_state_dict(load_dict['optimizer'])
        print(f"Loaded checkpoint from {to_absolute_path(cfg.load_path)}\n")

    for i in range(cfg.max_steps):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        batch = batch.to(device)
        loss = vae.loss_function(*vae.forward(batch), **{'M_N': cfg.KLD_weight,
                                                         'recons_loss': cfg.recons_loss,
                                                         'var_weight': cfg.var_weight})
        sample_from_batch = vae.sample_from_image(batch)
        mse = F.mse_loss(batch, sample_from_batch).item()
        print(f"Step {i}: loss = {loss['loss']:0.6f} | "
              f"Reconstruction part = {loss['Reconstruction_Loss']:0.6f} | "
              f"KLD part = {loss['KLD']:0.6f} | "
              f"MSE = {mse:0.6f} | "
              f"MAE = {(torch.abs(batch - sample_from_batch)).mean():0.6f} | "
              f"var = {loss['var']:0.6f}")

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        if i % cfg.save_every_steps == 0 and i > 0:
            vae.save(path=to_absolute_path(cfg.save_path), optimizer=optimizer)

    if cfg.sample_after_training:
        f, axarr = plt.subplots(3, cfg.n_samples)
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        sample_from_batch = vae.sample_from_image(batch)
        # sample_from_batch = vae.generate(batch)

        randomly_generated = []
        for i in range(min(cfg.n_samples, cfg.batch_size)):
            sample = sample_from_batch[i].detach().numpy()
            image = batch[i].detach().numpy()
            image = np.moveaxis(image, 0, -1)
            sample = np.moveaxis(sample, 0, -1)
            axarr[0, i].imshow(normalize_image(image * 255))
            axarr[1, i].imshow(normalize_image(sample * 255))
            axarr[1, i].set_xlabel(f"MSE = {((sample - image)**2).mean():0.4f}, "
                                   f"MAE = {(np.abs(sample - image)).mean():0.4f}")

            generated = vae.sample(1, device).detach().numpy()[0]
            generated = np.moveaxis(generated, 0, -1)
            randomly_generated.append(generated)
            if i == min(cfg.n_samples, cfg.batch_size) - 1:
                diff = (randomly_generated[0] - randomly_generated[1]) > 0.01
                result = np.zeros(diff.shape, dtype='uint8')
                diff_val = 100
                result[diff] = diff_val
                w, h, _ = diff.shape
                for m in range(w):
                    for n in range(h):
                        if any(result[m, n] == diff_val):
                            result[m, n, 0] = 0
                            result[m, n, 1] = diff_val
                            result[m, n, 2] = 0
                axarr[2, i].imshow(normalize_image(result + randomly_generated[0] * 255))
            else:
                axarr[2, i].imshow(normalize_image(generated * 255))
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
