from hydra.utils import to_absolute_path
import os
import numpy as np
import random
from PIL import Image


class BatchGenerator:
    def __init__(self, image_dir, batch_size=64, n_batches=10):
        self.batch_size = batch_size
        self.n_batches = n_batches

        image_dir = to_absolute_path(image_dir)
        files = os.listdir(image_dir)
        image_names = np.array(
            list(filter(lambda file: file.endswith('.jpg') or
                                     file.endswith('.png') or
                                     file.endswith('.bmp'), files)))
        all_images = np.vectorize(lambda img: f"{image_dir}/{img}")(image_names)
        all_images.sort()
        self.image_names = all_images

        self.loaded_images = []

    def get_batch(self):
        if len(self.loaded_images) < self.batch_size:
            self._load_images()

        batch = np.array([self.loaded_images.pop() for _ in range(self.batch_size)])
        # [B x W x H x C] -> [B x C x W x H]
        batch = np.rollaxis(batch, 3, 1)
        return batch

    def _load_images(self):
        k = min(self.batch_size * self.n_batches, len(self.image_names))
        images_to_load = random.choices(self.image_names, k=k)
        del self.loaded_images
        self.loaded_images = [np.array(Image.open(image)) for image in images_to_load]
