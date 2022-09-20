import torch
from CBIR.models import VAE, AutoEncoder
from CBIR.models import BatchGenerator
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from CBIR.kernel.utils import normalize_image
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize


def fit_VAE(cfg):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    if cfg.use_AE:
        model = AutoEncoder(cfg.input_size, cfg.in_channels,
                            cfg.lattent_dims, hidden_dims, cfg.n_conv_layers)
    else:
        model = VAE(cfg.input_size, cfg.in_channels,
                    cfg.lattent_dims, hidden_dims, cfg.n_conv_layers)
    model.move_to_device(device)

    print_train_info(cfg)

    model.summary()

    batch_generator = BatchGenerator(cfg.image_dir, batch_size=cfg.batch_size,
                                     skip_background=cfg.skip_background,
                                     use_MNIST=cfg.use_MNIST, input_size=cfg.input_size)
    params = model.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    if cfg.load_checkpoint:
        load_dict = model.load(to_absolute_path(cfg.load_path))
        optimizer.load_state_dict(load_dict['optimizer'])
        print(f"Loaded checkpoint from {to_absolute_path(cfg.load_path)}\n")

    experiment_name = f'{cfg.hidden_dims}_' \
                      f'lattent_dims{cfg.lattent_dims}_' \
                      f'lr{cfg.lr}'
    writer = SummaryWriter(to_absolute_path(f'{cfg.summarywriter_logdir}/{experiment_name}'))

    for i in range(cfg.max_steps + 1):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        batch = batch.to(device)
        loss = model.loss_function(*model.forward(batch), **{'M_N': cfg.KLD_weight,
                                                         'recons_loss': cfg.recons_loss,
                                                         'var_weight': cfg.var_weight})
        # sample_from_batch = vae.sample_from_image(batch)
        sample_from_batch = model.generate(batch)
        mse = F.mse_loss(batch, sample_from_batch).item()
        print(f"Step {i}: loss = {loss['loss']:0.6f} | "
              f"Reconstruction part = {loss['Reconstruction_Loss']:0.6f} | "
              f"KLD part = {loss['KLD']:0.6f} | "
              f"MSE = {mse:0.6f} | "
              f"MAE = {(torch.abs(batch - sample_from_batch)).mean():0.6f} | "
              f"var = {loss['var']:0.6f}")

        writer.add_scalar('loss', loss['loss'], i)
        writer.add_scalar('Reconstruction part', loss['Reconstruction_Loss'], i)
        writer.add_scalar('KLD part', loss['KLD'], i)
        writer.add_scalar('MSE', mse, i)
        writer.flush()

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        if i % cfg.save_every_steps == 0 and i > 0:
            model.save(path=to_absolute_path(cfg.save_path), optimizer=optimizer)

    if cfg.sample_after_training:
        sample_after_training(cfg=cfg, vae=model, device=device, batch_generator=batch_generator)


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


def sample_after_training(cfg, vae, batch_generator, device):
    f, axarr = plt.subplots(3, cfg.n_samples)
    f.set_size_inches(14, 10)
    for w in range(axarr.shape[0]):
        for h in range(axarr.shape[1]):
            axarr[w, h].axis('off')

    randomly_generated = []
    for i in range(cfg.n_samples):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        batch = batch.to(device)
        # sample_from_batch = vae.sample_from_image(batch)
        sample_from_batch = vae.generate(batch)

        sample = sample_from_batch[0].cpu().detach().numpy()
        image = batch[0].cpu().detach().numpy()
        image = np.moveaxis(image, 0, -1)
        sample = np.moveaxis(sample, 0, -1)
        axarr[0, i].imshow(normalize_image(image * 255))
        axarr[0, i].set_title(f"mean = {image.mean():0.3f}, var = {image.var():0.3f}")
        axarr[1, i].imshow(normalize_image(sample * 255))
        axarr[1, i].set_title(f"MSE = {((sample - image) ** 2).mean():0.4f}, "
                              f"MAE = {(np.abs(sample - image)).mean():0.4f}")

        generated = vae.sample(1, device).cpu().detach().numpy()[0]
        generated = np.moveaxis(generated, 0, -1)
        randomly_generated.append(generated)
        if i == cfg.n_samples - 1:
            diff = np.abs(randomly_generated[0] - randomly_generated[1]) > 0.01
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
            axarr[2, i].set_title(f"Diff between generated 1 and 2")
        else:
            axarr[2, i].imshow(normalize_image(generated * 255))
    name = f"{cfg.recons_loss}_loss_" \
           f"{cfg.lattent_dims}_lattent_" \
           f"{cfg.max_steps}_steps_" \
           f"{cfg.hidden_dims}_conv_" \
           f"{cfg.n_conv_layers}_layers_" \
           f"{cfg.input_size}_size_" \
           f"{cfg.lr}_lr_" \
           f"{cfg.batch_size}_batch_size"
    f.suptitle(name)
    plt.savefig(to_absolute_path(f"./eval_results/runs/{name}.jpg"))
    # plt.show()
