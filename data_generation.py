from hydra.utils import to_absolute_path
import numpy as np
from openslide import OpenSlide


def data_generation(cfg):
    print(f"{cfg.n_images}")
