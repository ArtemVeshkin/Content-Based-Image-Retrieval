from hydra.utils import to_absolute_path
import os
import numpy as np
import random
from PIL import Image
from torchvision.datasets import MNIST
from skimage.transform import resize
import matplotlib.pyplot as plt


class BatchGenerator:
    def __init__(self, image_dir, batch_size=64, n_batches=1,
                 skip_background=False, use_MNIST=False, input_size=224):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.skip_background = skip_background
        self.use_MNIST = use_MNIST
        self.MNIST_data = None
        self.input_size = input_size

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
        if self.use_MNIST:
            if self.MNIST_data is None:
                print(f"=====LOADING MNIST=====")
                self.MNIST_data = MNIST(
                    root='data',
                    train=True,
                    download=True
                )
            im_size = self.input_size
            batch = np.array([
                np.reshape(
                    np.tile(
                        resize(random.choice(self.MNIST_data.data).numpy(), (im_size, im_size)),
                        (3, 1)),
                    (3, im_size, im_size))
                for _ in range(self.batch_size)])

            return batch

        if len(self.loaded_images) < self.batch_size:
            self._load_images()

        batch = np.array([self.loaded_images.pop() for _ in range(self.batch_size)])
        # [B x W x H x C] -> [B x C x W x H]
        batch = np.rollaxis(batch, 3, 1)
        return batch

    def _load_images(self):
        k = min(self.batch_size * self.n_batches, len(self.image_names))
        del self.loaded_images
        loaded = 0
        im_size = self.input_size
        self.loaded_images = []
        while loaded < k:
            image = random.choice(self.image_names)
            image = resize(np.array(Image.open(image)), (im_size, im_size))
            if self.skip_background:
                if image.mean() > 0.9:
                    continue
            self.loaded_images.append(image)
            loaded += 1
