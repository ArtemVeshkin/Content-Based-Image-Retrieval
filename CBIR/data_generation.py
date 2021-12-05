from PIL import Image
from hydra.utils import to_absolute_path
import numpy as np
import os
from openslide import OpenSlide
import random
from tqdm import tqdm


def data_generation(cfg):
    image_dir = to_absolute_path(cfg.image_dir)
    files = os.listdir(image_dir)
    image_names = np.array(
        list(filter(lambda x: x.endswith('.svs') and x not in cfg.skip_images, files)))
    all_images = np.vectorize(lambda x: f"{image_dir}/{x}")(image_names)
    all_images.sort()
    wsi_list = [OpenSlide(image_name) for image_name in all_images]

    print(f'=====GENERATING TILES======')
    for i in tqdm(range(cfg.n_images)):
        tile: Image = generate_tile(wsi_list, cfg.level, cfg.tile_size)
        tile = tile.convert('RGB')
        output_name = f"{to_absolute_path(cfg.output_dir)}/{i}.jpg"
        tile.save(output_name)
    print(f'====={cfg.n_images} TILES GENERATED======')


def generate_tile(wsi_list, level, tile_size):
    wsi = random.choice(wsi_list)
    location = (
        random.randint(0, (wsi.dimensions[0] - tile_size[0]) // (level + 1)),
        random.randint(0, (wsi.dimensions[1] - tile_size[1]) // (level + 1))
    )
    tile = wsi.read_region(location, level, tile_size)
    return tile
