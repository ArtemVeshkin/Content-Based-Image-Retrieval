import torch
from torchsummary import summary
from models import VAE
from models import BatchGenerator


def fit_VAE(cfg):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    print_train_info(cfg)

    hidden_dims = cfg.hidden_dims if len(cfg.hidden_dims) > 0 else None
    vae = VAE(cfg.input_size, cfg.in_channels,
              cfg.lattent_dims, hidden_dims)
    vae.move_to_device(device)

    batch_generator = BatchGenerator(cfg.image_dir, batch_size=cfg.batch_size)
    params = vae.get_params()
    optimizer = torch.optim.Adam(params=params, lr=cfg.lr)
    for i in range(cfg.max_steps):
        batch = batch_generator.get_batch()
        batch = torch.FloatTensor(batch)
        batch = batch.to(device)
        loss = vae.loss_function(*vae.forward(batch), **{'M_N': 1})
        print(f"Step {i}: loss = {loss['loss']:0.4f} | "
              f"Reconstruction part = {loss['Reconstruction_Loss']:0.4f} | "
              f"KLD part = {loss['KLD']:0.4f}")

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()


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
