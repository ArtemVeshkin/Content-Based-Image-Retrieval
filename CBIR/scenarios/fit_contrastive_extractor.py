import random
import slideio
from hydra.utils import to_absolute_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import imgaug.augmenters as iaa
import torch
from CBIR.models import ContrastiveExtractor
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy, cosine_embedding_loss
from torch.utils.tensorboard import SummaryWriter
import gc


wsi_list = []


def fit_contrastive_extractor(cfg):
    init_wsi_list(cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = get_tensorboard_writer(cfg)

    model, optimizer = get_model(cfg)
    model = model.to(device)
    model.summary()

    if cfg.load_from_checkpoint:
        model.load(to_absolute_path(cfg.checkpoint_path))
        model.to(device)
        print(f'Model state loaded from {cfg.checkpoint_path}')

    train_loss = 0.
    train_metric = 0.
    for step in range(cfg.n_steps):
        x, y = generate_batch(cfg)
        x1, x2, y = x[0].to(device), x[1].to(device), y.to(device)
        out1, out2 = model(x1, x2)
        # loss = binary_cross_entropy(pred, y)
        # loss = calc_loss_cos(out1=out1, out2=out2, y=y, cfg=cfg, device=device)
        loss = calc_euclidean_loss(out1=out1, out2=out2, y=y, cfg=cfg, device=device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            train_loss += loss.item()
            # train_metric += accuracy(pred, y).item()
            # train_metric += cosine_accuracy(out1, out2, y)
            train_metric += roc_auc(out1, out2, y, cfg=cfg, device=device)
            # train_metric += small_values_ratio(cfg=cfg, out1=out1, out2=out2, device=device)
            if step % cfg.log_every_steps == 0:
                log(writer=writer, step=step, loss=train_loss, metric=train_metric,
                    averaging=cfg.log_every_steps if step != 0 else 1, mode='train')
                train_loss = 0.
                train_metric = 0.

        if step % cfg.save_every_steps == 0:
            model.save(to_absolute_path(cfg.checkpoint_path), optimizer)


def get_model(cfg):
    model = ContrastiveExtractor(embedding_size=cfg.embedding_size)

    lr = 1e-3 if 'lr' not in cfg else cfg.lr
    weight_decay = 0 if 'weight_decay' not in cfg else cfg.weight_decay

    optimizer = Adam(params=model.parameters(),
                     lr=lr,
                     weight_decay=weight_decay)
    return model, optimizer


def calc_loss_cos(out1, out2, y, cfg, device):
    ones = torch.ones(cfg.batch_size, cfg.embedding_size).to(device)
    zeros = torch.zeros(cfg.batch_size, cfg.embedding_size).to(device)
    out1_small_values_penalty = torch.mean(torch.where(torch.abs(out1) < cfg.values_penalty, ones, zeros))
    out2_small_values_penalty = torch.mean(torch.where(torch.abs(out2) < cfg.values_penalty, ones, zeros))
    small_values_penalty = cfg.small_values_penalty_alpha * (10 / cfg.batch_size) * \
                           (out1_small_values_penalty + out2_small_values_penalty)
    return cosine_embedding_loss(out1, out2, y, margin=cfg.cos_margin) + small_values_penalty


def calc_euclidean_loss(out1, out2, y, cfg, device):
    # D = torch.sqrt(((out1 - out2) ** 2).sum(dim=1))
    D = (torch.abs(out1 - out2)).sum(dim=1)
    loss = y * D + (1 - y) * torch.maximum(torch.Tensor([0]).to(device), cfg.euclidean_loss_margin - D) ** 2
    # print(loss, D, y)
    return loss.mean()


def accuracy(pred, target):
    return (torch.round(pred) == target).float().mean()


def roc_auc(out1, out2, target, cfg, device):
    from ignite.contrib.metrics import ROC_AUC
    from ignite.engine import Engine

    def eval_step(engine, batch):
        return batch
    print()
    default_evaluator = Engine(eval_step)

    D = torch.sqrt(((out1 - out2) ** 2).sum(dim=1))
    roc_auc = ROC_AUC()
    roc_auc.attach(default_evaluator, 'roc_auc')
    state = default_evaluator.run([[D, 1 - target]])
    print()
    return state.metrics['roc_auc']


def cosine_accuracy(embedding1, embedding2, target):
    # pred = normalize(torch.cosine_similarity(embedding1, embedding2), -1, 1)
    pred = torch.cosine_similarity(embedding1, embedding2)
    # target = normalize(target[:, None], -1, 1)
    target = target[:, None]
    return accuracy(pred, target).item()


def small_values_ratio(cfg, out1, out2, device):
    ones = torch.ones(cfg.batch_size, cfg.embedding_size).to(device)
    zeros = torch.zeros(cfg.batch_size, cfg.embedding_size).to(device)
    out1_small_values_penalty = torch.mean(torch.where(torch.abs(out1) < cfg.values_penalty, ones, zeros))
    out2_small_values_penalty = torch.mean(torch.where(torch.abs(out2) < cfg.values_penalty, ones, zeros))
    return ((out1_small_values_penalty + out2_small_values_penalty) / 2).item()


def normalize(tensor, min, max):
    return (tensor + min) / (max - min)


def get_tensorboard_writer(cfg):
    # experiment_name = f'{cfg.conv_hidden_dims}_' \
    #                   f'{cfg.fc_hidden_dims}_' \
    #                   f'conv_out_size{cfg.conv_out_size}_' \
    #                   f'lr{cfg.lr}'
    experiment_name = f'lr={cfg.lr}_' \
                      f'emb_size={cfg.embedding_size}_' \
                      f'small_abs_' \
                      f'margin={cfg.euclidean_loss_margin}'
                      # f'cos_margin={cfg.cos_margin}_'

    writer = SummaryWriter(to_absolute_path(f'{cfg.summarywriter_logdir}/{experiment_name}'))

    layout = {'Accuracy': {'train vs eval': ['Multiline', ['train/accuracy', 'eval/accuracy']]},
              'Loss': {'train vs eval': ['Multiline', ['train/loss', 'eval/loss']]}}
    writer.add_custom_scalars(layout)
    return writer


def log(writer, step, loss, metric, averaging=1, mode='train'):
    loss /= averaging
    metric /= averaging
    print(f'Step: {step} | Loss: {loss:0.4f} | Metric: {metric:0.4f}')
    writer.add_scalar(f'{mode}/loss', loss, step)
    writer.add_scalar(f'{mode}/metric', metric, step)
    writer.flush()


def show_images(images_list):
    f, axarr = plt.subplots(1, len(images_list))
    for i, image in enumerate(images_list):
        axarr[i].imshow(image)
    plt.show()


# X = [batch_size x W x H x channels * 2]
# Y = [batch_size] (0 / 1)
def generate_batch(cfg):
    x, y = [], []
    pool = Pool()
    for i in range(cfg.batch_size):
        # same [y = 1]
        if random.choice([True, False]):
            y.append(1)
            tiles = pool.apply_async(gen_same_tiles, [cfg])
            x.append(tiles)

        # different [y = -1]
        else:
            y.append(0)
            tiles = pool.apply_async(gen_different_tiles, [cfg])
            x.append(tiles)

    x = list(map(lambda tile: tile.get(), x))

    x1 = torch.moveaxis(torch.stack(list(map(lambda v: torch.Tensor(v[0]), x))), 3, 1)
    x2 = torch.moveaxis(torch.stack(list(map(lambda v: torch.Tensor(v[1]), x))), 3, 1)
    # y = torch.Tensor(y)[:, None]
    y = torch.Tensor(y)
    return (x1, x2), y


def gen_same_tiles(cfg):
    tile, _ = sample_tile(cfg)

    tile1 = apply_augmentation(cfg.augmentation, tile)
    tile2 = apply_augmentation(cfg.augmentation, tile)
    return tile1, tile2


def gen_different_tiles(cfg):
    tile1, meta = sample_tile(cfg)
    tile2 = None
    # nearby tile
    if random.choices([True, False], weights=[0.75, 0.25]):
        tile2 = get_nearby_tile(cfg, wsi=meta['wsi'], location=meta['location'])
    # random tile
    else:
        tile2, meta = sample_tile(cfg)

    tile1 = apply_augmentation(cfg.augmentation, tile1)
    tile2 = apply_augmentation(cfg.augmentation, tile2)
    return tile1, tile2


def sample_tile(cfg):
    generated = False
    tile = location = wsi = None
    tile_size = cfg.tile_size
    level = cfg.level
    while not generated:
        wsi = random.choice(wsi_list)
        location = (
            random.randint(0, wsi.size[0] - tile_size * (2 ** level)),
            random.randint(0, wsi.size[1] - tile_size * (2 ** level))
        )

        try:
            tile = wsi.read_block(rect=(*location,
                                        *([tile_size * (2 ** level)] * 2)),
                                  size=([tile_size] * 2))
        except Exception as e:
            continue

        if cfg.filter_background and (tile.var() <= 20. or tile.mean() >= 238):
            continue
        generated = True

    return tile, {
        'location': location,
        'wsi': wsi,
    }


def get_nearby_tile(cfg, wsi, location):
    generated = False
    tile = None
    tile_size = cfg.tile_size
    level = cfg.level
    while not generated:
        w_location = random.choice([location[0] - tile_size * (2 ** level),
                                    location[0],
                                    location[0] + tile_size * (2 ** level)])

        h_location = random.choice([location[1] - tile_size * (2 ** level),
                                    location[1],
                                    location[1] + tile_size * (2 ** level)])
        if (w_location, h_location) == location:
            continue

        try:
            tile = wsi.read_block(rect=(w_location, h_location,
                                        *([tile_size * (2 ** level)] * 2)),
                                  size=([tile_size] * 2))
        except Exception as e:
            continue
        generated = True

    return tile


def apply_augmentation(cfg, tile):
    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontally flip
            iaa.Flipud(0.5),  # vertically flip
            iaa.SomeOf(1, [
                iaa.AdditiveGaussianNoise(scale=(0, cfg.gaussian_noise_max * 255), per_channel=True),
                iaa.AdditiveGaussianNoise(scale=(0, cfg.gaussian_noise_max * 255), per_channel=False),
            ]),
            sometimes(iaa.ElasticTransformation(alpha=(cfg.elastic_transform_min * 10, cfg.elastic_transform_max * 10),
                                                sigma=(cfg.elastic_transform_min, cfg.elastic_transform_max),
                                                mode='reflect')),
            iaa.Affine(rotate=(-cfg.rotate_angle, cfg.rotate_angle), mode='reflect'),
            iaa.Affine(scale=(1. - cfg.scale, 1. + cfg.scale), mode='reflect'),
            iaa.GammaContrast((cfg.contrast_min, cfg.contrast_max)),
            iaa.MultiplyHueAndSaturation((1. - cfg.hue_and_saturation_delta,
                                          1. + cfg.hue_and_saturation_delta), per_channel=True),

        ], random_order=True
    )

    tile_aug = seq(image=tile)
    return tile_aug


def init_wsi_list(cfg):
    global wsi_list
    print(f"Loading WSI from paths:")
    wsi_list = []
    for wsi_path in cfg.wsi_paths:
        print(f'\t{wsi_path}', end=' ')
        wsi = slideio.open_slide(to_absolute_path(wsi_path), 'SVS').get_scene(0)
        if wsi.magnification == cfg.wsi_scale:
            wsi_list.append(wsi)
            print('[LOADED]')
        else:
            print('[WSI SCALE MISSMATCH]')
