from PIL import Image
from hydra.utils import to_absolute_path
import numpy as np
import os
import matplotlib.pyplot as plt


def crop_sources(cfg):
    cropped_dir = to_absolute_path(cfg.cropped_dir)
    out_size = cfg.out_size

    for cur_dir in cfg.sources_dir:
        files = os.listdir(cur_dir)
        image_names = np.array(
            list(filter(lambda x: x.endswith(('.jpg', '.png')), files)))
        all_images = np.vectorize(lambda x: f"{cur_dir}/{x}")(image_names)

        cropped_idx = 0
        for image_path in all_images:
            image = np.array(Image.open(image_path))

            if len(image.shape) != 3:
                continue

            w, h = image.shape[:2]
            if w < out_size * 3 or h < out_size * 3:
                continue

            for i in range(w // out_size):
                for j in range(h // out_size):
                    cropped = image[i * out_size:(i + 1) * out_size,
                                    j * out_size:(j + 1) * out_size, :]

                    Image.fromarray(cropped).save(f'{cropped_dir}/{cropped_idx}.jpg')
                    cropped_idx += 1
