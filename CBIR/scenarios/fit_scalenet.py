import torch
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from CBIR.models import ScaleNet, BatchGenerator
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import to_absolute_path
from tqdm import tqdm


def fit_scalenet(cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = get_tensorboard_writer(cfg)

    # train and validation data
    train_data, eval_data = get_data_generators(cfg)

    # model
    model, optimizer = get_model(cfg)
    if cfg.load_from_checkpoint:
        load_dict = model.load(to_absolute_path(cfg.checkpoint_path))
        # optimizer.load_state_dict(load_dict['optimizer'])
        print(f'Loaded checkpoint from {to_absolute_path(cfg.checkpoint_path)}')
    model.to(device)
    model.summary()

    # fit
    train_loss = 0.
    train_accuracy = 0.
    for step in range(cfg.training_steps):
        x_train, y_train = get_data(train_data, device)
        pred = model(x_train)
        loss = binary_cross_entropy(pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_accuracy += accuracy(pred, y_train).item()
        if step % cfg.log_every_steps == 0:
            log(writer=writer, step=step, loss=train_loss, accuracy=train_accuracy,
                averaging=cfg.log_every_steps if step != 0 else 1, mode='train')
            train_loss = 0.
            train_accuracy = 0.

        if step % cfg.eval_every_steps == 0:
            eval_scalenet(cfg=cfg, step=step, model=model, eval_data=eval_data,
                          device=device, writer=writer)

        if step % cfg.save_every_steps == 0:
            model.save(to_absolute_path(cfg.checkpoint_path), optimizer)


def eval_scalenet(cfg, step, model, eval_data, device, writer):
    print(f'EVALUATING MODEL ON {cfg.eval_num_steps * cfg.batch_size} IMAGES')
    with torch.inference_mode():
        eval_loss = 0.
        eval_accuracy = 0.
        for _ in tqdm(range(cfg.eval_num_steps)):
            x_eval, y_eval = get_data(eval_data, device)
            pred = model(x_eval)
            eval_loss += binary_cross_entropy(pred, y_eval).item()
            eval_accuracy += accuracy(pred, y_eval)
        log(writer=writer, step=step, loss=eval_loss, accuracy=eval_accuracy,
            averaging=cfg.eval_num_steps, mode='eval')


def log(writer, step, loss, accuracy, averaging=1, mode='train'):
    loss /= averaging
    accuracy /= averaging
    print(f'Step: {step} | Loss: {loss:0.4f} | Accuracy: {accuracy:0.4f}')
    writer.add_scalar(f'{mode}/loss', loss, step)
    writer.add_scalar(f'{mode}/accuracy', accuracy, step)
    writer.flush()


def get_tensorboard_writer(cfg):
    experiment_name = f'{cfg.conv_hidden_dims}_' \
                      f'{cfg.fc_hidden_dims}_' \
                      f'conv_out_size{cfg.conv_out_size}_' \
                      f'lr{cfg.lr}'
    writer = SummaryWriter(to_absolute_path(f'{cfg.summarywriter_logdir}/{experiment_name}'))

    layout = {'Accuracy': {'train vs eval': ['Multiline', ['train/accuracy', 'eval/accuracy']]},
              'Loss': {'train vs eval': ['Multiline', ['train/loss', 'eval/loss']]}}
    writer.add_custom_scalars(layout)
    return writer


def accuracy(pred, target):
    return (torch.round(pred) == target).float().mean()


def get_data_generators(cfg):
    train_generator = BatchGenerator(image_dirs=cfg.train_path,
                                     batch_size=cfg.batch_size * 2,
                                     n_batches=cfg.n_batches,
                                     input_size=cfg.input_size,
                                     give_image_names=True)
    eval_generator = BatchGenerator(image_dirs=cfg.eval_path,
                                    batch_size=cfg.batch_size * 2,
                                    n_batches=cfg.n_batches,
                                    input_size=cfg.input_size,
                                    give_image_names=True)
    print(f'Batch Generators are ready')
    return train_generator, eval_generator


def generate_scalenet_batch(image_batch, names_batch):
    batch_size = image_batch.shape[0]

    names_batch = np.vectorize(lambda path: os.path.basename(path))(names_batch)
    scales_batch = np.vectorize(lambda name: name[1:name.find('_')])(names_batch)

    image_batch_1_half = image_batch[:batch_size // 2, ...]
    image_batch_2_half = image_batch[batch_size // 2:, ...]

    scales_batch_1_half = scales_batch[:batch_size // 2, ...]
    scales_batch_2_half = scales_batch[batch_size // 2:, ...]

    second_half_scales = {}
    for i, scale in enumerate(scales_batch_2_half):
        if scale not in second_half_scales:
            second_half_scales[scale] = []
        second_half_scales[scale].append(image_batch_2_half[i])

    data_shape = list(image_batch_1_half.shape)
    data_shape[1] *= 2
    data = np.empty(data_shape)
    target = np.empty(data_shape[:1])
    for i in range(batch_size // 2):
        cur_scale = scales_batch_1_half[i]
        # to prevent class disbalance
        scale_candidates = list(second_half_scales.keys())
        if cur_scale in scale_candidates:
            scale_candidates += [cur_scale] * (len(scale_candidates) - 1)
        second_scale = random.choice(scale_candidates)
        second_image = random.choice(second_half_scales[second_scale])

        image_pair = np.concatenate((image_batch_1_half[i], second_image))
        data[i, ...] = image_pair

        target[i] = 1. if cur_scale == second_scale else 0.
    target = np.expand_dims(target, axis=1)
    return data, target


def get_data(batch_generator: BatchGenerator, device: torch.device):
    x_data, y_data = generate_scalenet_batch(*batch_generator.get_batch())
    x_data = torch.FloatTensor(x_data).to(device)
    y_data = torch.FloatTensor(y_data).to(device)
    return x_data, y_data


def get_model(cfg):
    model = ScaleNet(input_size=cfg.input_size,
                     grayscale_input=cfg.grayscale_input,
                     conv_hidden_dims=cfg.conv_hidden_dims,
                     conv_out_size=cfg.conv_out_size,
                     fc_hidden_dims=cfg.fc_hidden_dims)

    lr = 1e-3 if 'lr' not in cfg else cfg.lr
    weight_decay = 0 if 'weight_decay' not in cfg else cfg.weight_decay

    optimizer = Adam(params=model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    return model, optimizer
