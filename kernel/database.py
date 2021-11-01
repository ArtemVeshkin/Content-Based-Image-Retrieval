from kernel.utils import LSH, normalize_image, d_near
from kernel.extractors import EXTRACTORS

from tqdm import tqdm

from random import shuffle
import numpy as np
import pandas as pd
from PIL import Image
import os
from hydra.utils import to_absolute_path


class DataBase:
    def __init__(self, cfg, name: str = 'default'):
        self.name = name
        self.images = {}
        self.binary_features = {}
        self.cfg = cfg
        self.extractor = EXTRACTORS[cfg.feature_extractor.type](*cfg.feature_extractor.args)

        self.__tile_size = cfg.tile_size
        self.__hash = LSH(cfg.LSH_k_bits, cfg.feature_extractor.n_features)

    def load_images(self, directory: str, dataset_name: str, skip_and_return: int = 0):
        files = os.listdir(directory)
        image_names = np.array(
            list(filter(lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.bmp'), files)))
        all_images = np.vectorize(lambda x: directory + '/' + x)(image_names)
        all_images.sort()
        self.images[dataset_name] = all_images[skip_and_return:]
        return all_images[:skip_and_return]

    def get_image(self, dataset_name: str, image_idx: int):
        return np.array(Image.open(self.images[dataset_name][image_idx]))

    def extract_binary_features(self, dataset_name: str):
        dataset_images = self.images[dataset_name]
        self.binary_features[dataset_name] = np.zeros(dataset_images.shape[0], dtype=object)
        print(f"Extracting binary features for \"{dataset_name}\" dataset with "
              f"{self.cfg.feature_extractor.type} extractor:")
        for n, image in enumerate(tqdm(dataset_images)):
            image = np.array(Image.open(image))
            width, height = image.shape[0] // self.__tile_size, image.shape[1] // self.__tile_size
            binary_features = np.zeros((width, height))
            for i in range(width):
                for j in range(height):
                    tile = image[i * self.__tile_size:(i + 1) * self.__tile_size,
                           j * self.__tile_size:(j + 1) * self.__tile_size, :]
                    extracted_features = self.extractor.extract_features_for_tile(tile)
                    binary_features[i, j] = self.__hash.get_signature(extracted_features)
            self.binary_features[dataset_name][n] = binary_features

    def save_dataset_binary_features(self, dataset_name: str):
        d = {
            'binary_features': self.binary_features[dataset_name]
        }
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.transpose()
        df.to_pickle(self.cfg.binary_features_serialization + dataset_name)

    def load_dataset_binary_features(self, dataset_name: str):
        df = pd.read_pickle(self.cfg.binary_features_serialization + dataset_name)
        return df['binary_features'].values[0]

    def serialize(self, filename: str):
        d = {
            'Name': self.name,
            'Images': self.images,
            'Binary_codes': self.binary_features,

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
        self.binary_features = df['Binary_codes'].values[0]

        self.__tile_size = df['Tile_size'].values[0]
        self.__hash.k_bits = df['Hash_k_bits'].values[0]
        self.__hash.n_features = df['Hash_n_features'].values[0]
        self.__hash.seed = df['Hash_seed'].values[0]

    def search(self, filename: str, top_n: int):
        print(f"\nSearching top {top_n} similar patches for {filename}")
        image = np.array(Image.open(filename))
        width, height = image.shape[0] // self.__tile_size, image.shape[1] // self.__tile_size
        # Query image binarization
        query_binary_features = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                tile = image[i * self.__tile_size:(i + 1) * self.__tile_size,
                             j * self.__tile_size:(j + 1) * self.__tile_size, :]
                query_binary_features[i, j] = self.__hash.get_signature(
                    self.extractor.extract_features_for_tile(tile))
        query_binary_features = query_binary_features.ravel()

        c2_candidates = []
        dataset_candidates_count = {}
        for dataset_name, dataset_binary_features in self.binary_features.items():
            for image_idx in range(dataset_binary_features.shape[0]):
                for i in range(dataset_binary_features[image_idx].shape[0] - width + 1):
                    for j in range(dataset_binary_features[image_idx].shape[1] - height + 1):
                        if np.any(np.in1d(dataset_binary_features[image_idx][i:i + width, j:j + height],
                                          query_binary_features.ravel())):
                            if dataset_name not in dataset_candidates_count:
                                dataset_candidates_count[dataset_name] = 0
                            dataset_candidates_count[dataset_name] += 1
                            c2_candidates.append({
                                "dataset_name": dataset_name,
                                "image_idx": image_idx,
                                "x": i,
                                "y": j
                            })
        c2_candidates = np.array(c2_candidates)
        if c2_candidates.shape[0] == 0:
            print("No similar fragments in database:(")
            return
        print(f"{c2_candidates.shape[0]} candidates found:")
        for dataset in dataset_candidates_count.keys():
            print(f"{dataset_candidates_count[dataset]} in {dataset}")

        print(f"Calculating distances:")
        c2_distances_dict = {}
        for i in tqdm(range(c2_candidates.shape[0])):
            dataset_name = c2_candidates[i]["dataset_name"]
            image_idx = c2_candidates[i]["image_idx"]
            x = c2_candidates[i]["x"]
            y = c2_candidates[i]["y"]
            db_binary_features = self.binary_features[dataset_name][image_idx][x:x + width, y:y + height]
            distance = d_near(query_binary_features, db_binary_features) + \
                       d_near(db_binary_features, query_binary_features)
            if distance not in c2_distances_dict:
                c2_distances_dict[distance] = []
            c2_distances_dict[distance].append(c2_candidates[i])

        for distance in c2_distances_dict.keys():
            shuffle(c2_distances_dict[distance])

        used_tiles = {dataset_name: np.array([np.zeros(dataset[i].shape) for i in range(dataset.shape[0])])
                      for dataset_name, dataset in self.binary_features.items()}

        results = []
        n = 1
        distances_sorted = sorted(c2_distances_dict.keys())
        for distance in distances_sorted:
            if n > top_n:
                break
            for candidate in c2_distances_dict[distance]:
                dataset_name = candidate["dataset_name"]
                img_idx = candidate["image_idx"]
                img = self.get_image(dataset_name, img_idx)
                x = candidate["x"]
                y = candidate["y"]

                if (used_tiles[dataset_name][img_idx][x:x + width, y:y + height] == 0).all():
                    used_tiles[dataset_name][img_idx][x:x + width, y:y + height] = 1
                    candidate['distance'] = distance
                    results.append(candidate)
                    if self.cfg.save_results:
                        Image.fromarray(normalize_image(img[x * self.__tile_size: (x + width) * self.__tile_size,
                                                        y * self.__tile_size:(y + height) * self.__tile_size])) \
                            .save(to_absolute_path(f"Results/top_{n}.png"))
                    n += 1
                    if n > top_n:
                        break
        return np.array(results)
