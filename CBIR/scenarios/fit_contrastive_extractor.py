import os
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
from collections import Counter
from PIL import Image
from skimage.segmentation import felzenszwalb, mark_boundaries
from sklearn.metrics import roc_auc_score


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
    train_metric = []
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
        loss = loss.detach()

        with torch.no_grad():
            train_loss += loss.item()
            # train_metric += accuracy(pred, y).item()
            # train_metric += cosine_accuracy(out1, out2, y)
            # train_metric += roc_auc(out1, out2, y)
            train_metric += embedding_distances(out1, out2, y, cfg)
            # train_metric += small_values_ratio(cfg=cfg, out1=out1, out2=out2, device=device)
            if step % cfg.log_every_steps == 0:
                log(writer=writer, step=step, loss=train_loss, metric=train_metric,
                    averaging=cfg.log_every_steps if step != 0 else 1, mode='train')
                train_loss = 0.
                train_metric = []

        if step % cfg.save_every_steps == 0:
            model.save(to_absolute_path(cfg.checkpoint_path), optimizer)


def get_model(cfg):
    model = ContrastiveExtractor(embedding_size=cfg.embedding_size,
                                 conv_hidden_dims=cfg.conv_hidden_dims,
                                 fc_hidden_dims=cfg.fc_hidden_dims)

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


def roc_auc(out1, out2, target):
    d = torch.sqrt(((out1 - out2) ** 2).sum(dim=1)).cpu()
    return roc_auc_score(1 - target.cpu(), d)


def embedding_distances(out1, out2, target, cfg):
    d = torch.sqrt(((out1 - out2) ** 2).sum(dim=1))

    same_dist = d[target == 1].mean().cpu()
    near_dist = d[target == cfg.near_target].mean().cpu()
    diff_dist = d[target == 0].mean().cpu()

    return [same_dist, near_dist, diff_dist]


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
    experiment_name = f'lr={cfg.lr}_' \
                      f'emb_size={cfg.embedding_size}_' \
                      f'near_target={cfg.near_target}_' \
                      f'conv_hidden_dims={cfg.conv_hidden_dims}_' \
                      f'margin={cfg.euclidean_loss_margin}_' \
                      f'use_segm={cfg.use_segmentation}'

    writer = SummaryWriter(to_absolute_path(f'{cfg.summarywriter_logdir}/{experiment_name}'))

    layout = {'Metric': {'train vs eval': ['Multiline', [
        'train/metric', 'eval/metric',
        'train/same_dist', 'train/near_dist', 'train/diff_dist',
        'eval/same_dist', 'eval/near_dist', 'eval/diff_dist'
    ]]},
              'Loss': {'train vs eval': ['Multiline', ['train/loss', 'eval/loss']]}}
    writer.add_custom_scalars(layout)
    return writer


def log(writer, step, loss, metric, averaging=1, mode='train'):
    loss /= averaging
    if len(metric) == 3:
        metric = np.array(metric)
    metric /= averaging

    if len(metric) == 3:
        print(f'Step: {step} | Loss: {loss:0.4f} | '
              f'same_dist: {metric[0]:0.4f} | near_dist: {metric[1]:0.4f} | diff_dist: {metric[2]:0.4f}')
    else:
        print(f'Step: {step} | Loss: {loss:0.4f} | Metric: {metric:0.4f}')
    writer.add_scalar(f'{mode}/loss', loss, step)
    if len(metric) == 3:
        writer.add_scalar(f'{mode}/same_dist', metric[0], step)
        writer.add_scalar(f'{mode}/near_dist', metric[1], step)
        writer.add_scalar(f'{mode}/diff_dist', metric[2], step)
    else:
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
    samples = []
    pool = Pool()
    for i in range(cfg.batch_size):
        # same [y = 1]
        if random.choice([True, False]):
            # y.append(1)
            tiles = pool.apply_async(gen_same_tiles, [cfg])
            samples.append(tiles)

        # different [y = -1]
        else:
            # y.append(0)
            tiles = pool.apply_async(gen_different_tiles, [cfg])
            samples.append(tiles)

    samples = list(map(lambda tile: tile.get(), samples))

    x1 = torch.moveaxis(torch.stack(list(map(lambda v: torch.Tensor(v[0]), samples))), 3, 1)
    x2 = torch.moveaxis(torch.stack(list(map(lambda v: torch.Tensor(v[1]), samples))), 3, 1)
    y = torch.Tensor(list(map(lambda v: v[2], samples)))
    # y = torch.Tensor(y)[:, None]
    # y = torch.Tensor(y)
    return (x1, x2), y


def gen_same_tiles(cfg):
    tile, _ = sample_tile(cfg)

    tile1 = apply_augmentation(cfg.augmentation, tile)
    tile2 = apply_augmentation(cfg.augmentation, tile)

    return tile1, tile2, 1


def gen_different_tiles(cfg):

    tile1, meta = sample_tile(cfg)

    tile2 = None
    target = 0

    # nearby tile
    # if random.choices([True, False], weights=[0.75, 0.25]):
    if random.choice([True, False]):
        tile2, same_classes = get_nearby_tile(cfg, wsi=meta['wsi'], location=meta['location'])
        target = cfg.near_target
        if cfg.use_segmentation and same_classes:
            target = 1
    # random tile
    else:
        tile2, meta = sample_tile(cfg)

    tile1 = apply_augmentation(cfg.augmentation, tile1)
    tile2 = apply_augmentation(cfg.augmentation, tile2)
    return tile1, tile2, target


def sample_tile(cfg):
    generated = False
    tile = location = wsi = None
    tile_size = cfg.tile_size
    level = cfg.level
    while not generated:
        wsi = random.choice(wsi_list)

        if cfg.wsi_type == 'GDAL':
            h, w = wsi.shape[:2]
        else:
            h, w = wsi.size[:2]
        location = (
            random.randint(tile_size * (2 ** level), h - 2 * tile_size * (2 ** level)),
            random.randint(tile_size * (2 ** level), w - 2 * tile_size * (2 ** level))
        )
        try:
            if cfg.wsi_type == 'GDAL':
                tile = wsi[location[0]:location[0] + tile_size,
                           location[1]:location[1] + tile_size].copy()
            else:
                tile = wsi.read_block(rect=(*location,
                                            *([tile_size * (2 ** level)] * 2)),
                                      size=([tile_size] * 2))
        except RuntimeError as e:
            continue

        if cfg.filter_background and (tile.var() <= 20. or tile.mean() >= 238):
            continue
        generated = True

    return tile, {
        'location': location,
        'wsi': wsi,
    }


# returns most frequent class label in area
def get_area_class(area, x_min, x_max, y_min, y_max):
    area_fragment = area[x_min:x_max, y_min:y_max].ravel()
    return Counter(area_fragment).most_common(1)[0][0]


def get_nearby_tile(cfg, wsi, location):
    generated = False
    tile = None
    tile_size = cfg.tile_size
    level = cfg.level
    crop_size = tile_size * (2 ** level)

    candidate_area = wsi[max(0, location[0] - crop_size):
                         min(wsi.shape[0], location[0] + 2 * crop_size),
                         max(0, location[1] - crop_size):
                         min(wsi.shape[1], location[1] + 2 * crop_size), :]

    if cfg.use_segmentation:
        clustered_candidate_area = felzenszwalb(candidate_area, scale=1000, sigma=2, min_size=2000)
    else:
        clustered_candidate_area = np.zeros(candidate_area.shape[:2])

    source_class = get_area_class(clustered_candidate_area, x_min=crop_size, x_max=2 * crop_size,
                                                  y_min=crop_size, y_max=2 * crop_size)
    tile_class = source_class + 1

    while not generated:
        w_location = random.choice([location[0] - crop_size,
                                    location[0],
                                    location[0] + crop_size])

        h_location = random.choice([location[1] - crop_size,
                                    location[1],
                                    location[1] + crop_size])
        if (w_location, h_location) == location:
            continue
        try:
            if cfg.wsi_type == 'GDAL':
                tile = wsi[w_location:w_location + tile_size,
                           h_location:h_location + tile_size].copy()

                if tile.shape[:2] != (tile_size, tile_size):
                    raise RuntimeError()

                normalized_w = w_location - location[0] + crop_size
                normalized_h = h_location - location[1] + crop_size

                tile_class = get_area_class(clustered_candidate_area,
                                            x_min=normalized_w,
                                            x_max=normalized_w + tile_size,
                                            y_min=normalized_h,
                                            y_max=normalized_h + tile_size)
            else:
                tile = wsi.read_block(rect=(w_location, h_location,
                                            *([crop_size] * 2)),
                                      size=([tile_size] * 2))
        except RuntimeError:
            continue
        generated = True

    return tile, source_class == tile_class


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


def load_to_wsi_list(cfg, image_path):
    # jpeg/png/...
    if cfg.wsi_type == 'GDAL':
        image = np.array(Image.open(image_path))
        if len(image.shape) != 3:
            return False

        w, h = image.shape[:2]
        if w < cfg.tile_size * 3 or h < cfg.tile_size * 3:
            return False
        wsi_list.append(image)
        return True
    else:
        wsi = slideio.open_slide(to_absolute_path(image_path), cfg.wsi_type).get_scene(0)
        if wsi.magnification == cfg.wsi_scale:
            wsi_list.append(wsi)
            return True
        else:
            return False


def init_wsi_list(cfg):
    global wsi_list
    print(f"Loading WSI from paths:")
    wsi_list = []
    for wsi_path in cfg.wsi_paths:
        print(f'\t{wsi_path}', end=' ')

        result = True
        if wsi_path.endswith('/'):
            loaded = 0
            total = 0
            for image_path in os.listdir(wsi_path):
                image_path = f'{wsi_path}{image_path}'
                if load_to_wsi_list(cfg, image_path):
                    loaded += 1
                total += 1
            print(f'[LOADED {loaded}/{total}]')

        else:
            if load_to_wsi_list(cfg, wsi_path):
                print('[LOADED]')
            else:
                print('[WSI SCALE MISSMATCH]')

