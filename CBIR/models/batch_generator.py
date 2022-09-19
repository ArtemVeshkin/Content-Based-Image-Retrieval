from hydra.utils import to_absolute_path
import os
import numpy as np
import random
from PIL import Image
from torchvision.datasets import MNIST
from skimage.transform import resize
import matplotlib.pyplot as plt


class BatchGenerator:
    def __init__(self, image_dirs, batch_size=64, n_batches=1,
                 skip_background=False, use_MNIST=False, input_size=224, give_image_names=False):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.skip_background = skip_background
        self.use_MNIST = use_MNIST
        self.MNIST_data = None
        self.input_size = input_size
        self.give_image_names = give_image_names

        self.image_names = []
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        for image_dir in image_dirs:
            image_dir = to_absolute_path(image_dir)
            files = os.listdir(image_dir)
            image_names = np.array(
                list(filter(lambda file: file.endswith('.jpg') or
                                         file.endswith('.png') or
                                         file.endswith('.bmp') or
                                         file.endswith('.tif'), files)))
            all_images = np.vectorize(lambda img: f"{image_dir}/{img}")(image_names)
            all_images.sort()
            for image in all_images:
                self.image_names.append(image)

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

        batch = [self.loaded_images.pop() for _ in range(self.batch_size)]
        image_names = None
        if self.give_image_names:
            batch, image_names = zip(*batch)
            image_names = np.array(image_names)
        batch = np.array(batch)
        # [B x W x H x C] -> [B x C x W x H]
        batch = np.rollaxis(batch, 3, 1)
        if self.give_image_names:
            return batch, image_names
        return batch

    def _load_images(self):
        k = min(self.batch_size * self.n_batches, len(self.image_names))
        del self.loaded_images
        loaded = 0
        im_size = self.input_size
        self.loaded_images = []
        while loaded < k:
            image_name = random.choice(self.image_names)
            image = np.array(Image.open(image_name)) / 255

            if self.skip_background:
                if image.mean() > 0.9:
                    continue
            if image.shape[:2] != (im_size, im_size):
                if im_size > image.shape[0] or im_size > image.shape[1]:
                    image = resize(image, (im_size, im_size))
                else:
                    image = image[:im_size, :im_size, :]

            if self.give_image_names:
                image = (image, image_name)
            self.loaded_images.append(image)
            loaded += 1
