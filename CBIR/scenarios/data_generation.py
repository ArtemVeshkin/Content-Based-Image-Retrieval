from PIL import Image
from hydra.utils import to_absolute_path
import numpy as np
import os
import slideio
import random
from skimage.transform import resize
from tqdm import tqdm


def data_generation(cfg):
    image_dir = to_absolute_path(cfg.image_dir)
    files = os.listdir(image_dir)
    image_names = np.array(
        list(filter(lambda x: x.endswith('.svs') and x not in cfg.skip_images, files)))
    all_images = np.vectorize(lambda x: f"{image_dir}/{x}")(image_names)
    all_images.sort()
    wsi_list = [slideio.open_slide(image_name, 'SVS') for image_name in all_images]

    print(f'=====GENERATING TILES======')
    for i in tqdm(range(cfg.n_images)):
        tile = generate_tile(wsi_list, cfg.level, np.array(cfg.tile_size))
        tile = Image.fromarray(tile)
        tile = tile.convert('RGB')
        output_name = f"{to_absolute_path(cfg.output_dir)}/{i}.jpg"
        tile.save(output_name)
    print(f'====={cfg.n_images} TILES GENERATED======')


def generate_tile(wsi_list, level, tile_size):
    wsi = random.choice(wsi_list).get_scene(0)
    location = (
        random.randint(0, wsi.size[0] - tile_size[0] * (2 ** level)),
        random.randint(0, wsi.size[1] - tile_size[1] * (2 ** level))
    )
    tile = wsi.read_block(rect=(*location,
                                *(tile_size * (2 ** level))))
    tile = resize(tile, tile_size, preserve_range=True)
    tile = tile.astype('uint8')
    return tile
