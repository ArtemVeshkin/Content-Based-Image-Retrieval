import random
import slideio
from hydra.utils import to_absolute_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import imgaug.augmenters as iaa
import torch

wsi_list = []


def fit_contrastive_extractor(cfg):
    init_wsi_list(cfg)

    x, y = generate_batch(cfg)


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

        # different [y = 0]
        else:
            y.append(0)
            tiles = pool.apply_async(gen_different_tiles, [cfg])
            x.append(tiles)

    x = list(map(lambda tile: tile.get(), x))

    x1 = torch.stack(list(map(lambda v: v[0], x)))
    x2 = torch.stack(list(map(lambda v: v[1], x)))
    y = torch.Tensor(y)
    return (x1, x2), y


def gen_same_tiles(cfg):
    tile, _ = sample_tile(cfg)

    tile1 = apply_augmentation(cfg.augmentation, tile)
    tile2 = apply_augmentation(cfg.augmentation, tile)

    tile1 = torch.Tensor(tile1)
    tile2 = torch.Tensor(tile2)
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

    tile1 = torch.Tensor(tile1)
    tile2 = torch.Tensor(tile2)
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
