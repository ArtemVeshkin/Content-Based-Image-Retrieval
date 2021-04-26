from CBIR.utils import *

import numpy as np
import pandas as pd
from PIL import Image
import os
from tensorflow.keras import models, layers


class DataBase:
    def __init__(self, name: str = 'default'):
        self.name = name
        self.images = np.array([])
        self.features = np.array([], dtype=object)
        self.binary_codes = np.array([], dtype=object)
        self.model_path = None
        self.model = None

        self.__tile_size = 128
        self.__hash = LSH(32, 64)

    def load_images(self, directory: str):
        files = os.listdir(directory)
        image_names = np.array(
            list(filter(lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.bmp'), files)))
        self.images = np.vectorize(lambda x: directory + '/' + x)(image_names)

    def load_model(self, model_path: str):
        self.model_path = model_path
        self.model = models.load_model(model_path)
        for i in range(6):
            self.model.pop()
        self.model.add(layers.AveragePooling2D((2, 2)))

    def get_image(self, n: int):
        return np.array(Image.open(self.images[n]))

    def extract_features(self, tile_size: int):
        self.__tile_size = tile_size
        self.features = np.zeros(self.images.size, dtype=object)
        n = 0
        for image in self.images:
            print("Extracting features from ", n, " image")
            image = np.array(Image.open(image))
            width, height = image.shape[0] // tile_size, image.shape[1] // tile_size
            features = np.zeros((width, height), dtype=object)
            for i in range(width):
                for j in range(height):
                    tile = image[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size, :]
                    features[i, j] = get_nn_features(tile, self.model)
            self.features[n] = features
            n += 1

    def binarization(self):
        self.binary_codes = np.zeros(self.images.shape[0], dtype=object)
        for n in range(self.features.shape[0]):
            print("Binarization features of ", n, " image")
            width, height = self.features[n].shape
            binary_codes = np.zeros((width, height))
            for i in range(width):
                for j in range(height):
                    binary_codes[i, j] = self.__hash.get_signature(self.features[n][i, j])
            self.binary_codes[n] = binary_codes

    def serialize(self, filename: str):
        d = {
            'Name': self.name,
            'Images': self.images,
            'Features': self.features,
            'Binary_codes': self.binary_codes,
            'Model_path': self.model_path,
            'Tile_size': self.__tile_size,
            'Hash_k_bits': self.__hash.k_bits,
            'Hash_n_features': self.__hash.n_features,
            'Hash_seed': self.__hash.seed,
        }
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.transpose()

        df.to_pickle(filename)

    def deserialize(self, filename: str):
        df = pd.read_pickle(filename)

        self.name = df['Name'].values[0]
        self.images = df['Images'].values[0]
        self.features = df['Features'].values[0]
        self.binary_codes = df['Binary_codes'].values[0]
        self.model_path = df['Model_path'].values[0]
        self.load_model(self.model_path)

        self.__tile_size = df['Tile_size'].values[0]
        self.__hash.k_bits = df['Hash_k_bits'].values[0]
        self.__hash.n_features = df['Hash_n_features'].values[0]
        self.__hash.seed = df['Hash_seed'].values[0]

    def search(self, filename: str, top_n: int):
        image = np.array(Image.open(filename))
        width, height = image.shape[0] // self.__tile_size, image.shape[1] // self.__tile_size
        # Query image binarization
        binary_codes = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                tile = image[i * self.__tile_size:(i + 1) * self.__tile_size,
                       j * self.__tile_size:(j + 1) * self.__tile_size, :]
                binary_codes[i, j] = self.__hash.get_signature(get_nn_features(tile, self.model))
        binary_codes = binary_codes.ravel()

        c2_indices = []
        for n in range(self.binary_codes.shape[0]):
            for i in range(self.binary_codes[n].shape[0] - width + 1):
                for j in range(self.binary_codes[n].shape[1] - height + 1):
                    if np.any(np.in1d(self.binary_codes[n][i:i + width, j:j + height], binary_codes.ravel())):
                        c2_indices.append(np.array([n, i, j]))
        c2_indices = np.array(c2_indices)
        if c2_indices.shape[0] == 0:
            print("No similar fragments in database:(")
            return
        print(c2_indices.shape)

        c2_distances_dict = {}
        for i in range(c2_indices.shape[0]):
            print(i)
            n = c2_indices[i][0]
            x = c2_indices[i][1]
            y = c2_indices[i][2]
            distance = d_near(binary_codes, self.binary_codes[n][x:x + width, y:y + height]) + \
                       d_near(self.binary_codes[n][x:x + width, y:y + height], binary_codes)
            c2_distances_dict[distance] = c2_indices[i]

        used_tiles = np.zeros(self.binary_codes.shape, dtype=object)
        for i in range(self.binary_codes.shape[0]):
            used_tiles[i] = np.zeros(self.binary_codes[i].shape)

        n = 1
        keys_sorted = sorted(c2_distances_dict.keys())
        for k in keys_sorted:
            img = self.get_image(c2_distances_dict[k][0])
            x = c2_distances_dict[k][1]
            y = c2_distances_dict[k][2]

            if (used_tiles[c2_distances_dict[k][0]][x:x+width, y:y+height] == 0).all():
                used_tiles[c2_distances_dict[k][0]][x:x + width, y:y + height] = 1

                Image.fromarray(normalize_image(img[x * self.__tile_size: (x + width) * self.__tile_size,
                                                y * self.__tile_size:(y + height) * self.__tile_size])) \
                    .save("Results/top_" + str(n) + ".png")

                if n == top_n:
                    break
                n += 1


__all__ = ['DataBase']
