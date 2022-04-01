from CBIR.kernel.utils import LSH, normalize_image, d_near
from CBIR.kernel.extractors import EXTRACTORS
from CBIR.models import ScaleNet
from .search import *

from tqdm import tqdm

import random
import numpy as np
import pandas as pd
from PIL import Image
import os
from hydra.utils import to_absolute_path
from typing import List
import slideio


class DataBase:
    def __init__(self, cfg, name: str = 'default'):
        self.name = name

        # self.images = {dataset_name: {scale: [image_names]}}
        self.images = {}

        # self.features = {dataset_name: {
        #                       scale: [{'features'  : features,
        #                                'image_name': image_name}]
        #                       }
        #                  }
        self.features = {}
        self.cfg = cfg
        self.extractor = EXTRACTORS[cfg.feature_extractor.type](*cfg.feature_extractor.args)

        self.scalenet = ScaleNet(input_size=cfg.scalenet.input_size,
                                 grayscale_input=cfg.scalenet.grayscale_input,
                                 conv_hidden_dims=cfg.scalenet.conv_hidden_dims,
                                 conv_out_size=cfg.scalenet.conv_out_size,
                                 fc_hidden_dims=cfg.scalenet.fc_hidden_dims)
        self.scalenet.load(cfg.scalenet.checkpoint_path)

        self.__tile_size = cfg.tile_size
        self.__hash = LSH(cfg.LSH_k_bits, cfg.feature_extractor.n_features)

    def load_images(self, directory: str, dataset_name: str, skip_and_return: int = 0, scale: int = 0):
        files = os.listdir(directory)
        image_names = np.array(
            list(filter(lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.bmp'), files)))
        all_images = np.vectorize(lambda x: directory + '/' + x)(image_names)
        all_images.sort()
        if dataset_name not in self.images:
            self.images[dataset_name] = {}
        self.images[dataset_name][scale] = all_images[skip_and_return:]
        return all_images[:skip_and_return]

    def load_svs(self, path: str, dataset_name: str, scales: List[int]):
        if path.endswith('.svs') and os.path.exists(path):
            scales = np.array(scales)
            wsi = slideio.open_slide(path, 'SVS').get_scene(0)
            scales = scales[scales <= int(wsi.magnification)]

            if dataset_name not in self.images:
                self.images[dataset_name] = {}
            for scale in scales:
                if scale not in self.images[dataset_name]:
                    self.images[dataset_name][scale] = np.array([])
                self.images[dataset_name][scale] = np.append(self.images[dataset_name][scale], path)

    def get_image(self, dataset_name: str, scale: int, image_idx: int):
        image_name = self.images[dataset_name][scale][image_idx]
        if image_name.endswith('.svs'):
            wsi = slideio.open_slide(image_name, 'SVS').get_scene(0)
            cur_scale_size = (np.array(wsi.size) * (scale / wsi.magnification)).astype('int64')
            return wsi.read_block(size=cur_scale_size)
        else:
            return np.array(Image.open(image_name))

    # TODO: add smart and faster extraction (only for new images)
    def extract_features(self, dataset_name: str, scales=None):
        scales = self.images[dataset_name].keys() if scales is None else scales
        for s in scales:
            images = self.images[dataset_name][s]
            if dataset_name not in self.features:
                self.features[dataset_name] = {}
            self.features[dataset_name][s] = np.zeros(images.shape[0], dtype=object)
            print(f"Extracting binary features for \"{dataset_name}\" dataset on scale {s} "
                  f"with {self.cfg.feature_extractor.type} extractor:")
            for n, image_name in enumerate(tqdm(images, disable=images.shape[0] < 5)):
                if image_name.endswith('.svs'):
                    wsi = slideio.open_slide(image_name, 'SVS').get_scene(0)
                    cur_scale_size = (np.array(wsi.size) * (s / wsi.magnification)).astype('int64')
                    image = wsi.read_block(size=cur_scale_size)
                else:
                    image = np.array(Image.open(image_name))
                features = get_image_features(image=image, tile_size=self.__tile_size,
                                              extractor=self.extractor, binarizator=self.__hash,
                                              log=image_name.endswith('.svs'))
                self.features[dataset_name][s][n] = {
                    'features': features,
                    'image_name': image_name,
                }

    def save_dataset_features(self, dataset_name: str):
        d = {
            'features': self.features[dataset_name]
        }
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.transpose()
        df.to_pickle(self.cfg.features_serialization + dataset_name)

    def load_dataset_features(self, dataset_name: str):
        df = pd.read_pickle(self.cfg.features_serialization + dataset_name)
        return df['features'].values[0]

    def serialize(self, filename: str):
        d = {
            'Name': self.name,
            'Images': self.images,
            'Features': self.features,

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

        self.__tile_size = df['Tile_size'].values[0]
        self.__hash.k_bits = df['Hash_k_bits'].values[0]
        self.__hash.n_features = df['Hash_n_features'].values[0]
        self.__hash.seed = df['Hash_seed'].values[0]

    def search(self, image: np.ndarray, top_n: int, detect_scale=False, log=True):
        if log: print(f"\nSearching top {top_n} similar patches")
        width, height = image.shape[0] // self.__tile_size, image.shape[1] // self.__tile_size

        # Query image features
        query_features = get_image_features(image=image, tile_size=self.__tile_size,
                                            extractor=self.extractor, binarizator=self.__hash)

        # Query image scale detection
        query_scales = None
        if detect_scale:
            query_scales_dict = detect_scales(scalenet=self.scalenet,
                                              query=image,
                                              test_wsi=self.cfg.scale_detection.test_wsi,
                                              scales=self.cfg.scale_detection.scales,
                                              n_samples=self.cfg.scale_detection.n_samples)
            if log: print(f'Query is on scale {query_scales_dict["detected_scale"]} '
                          f'with a probability of {query_scales_dict["prob"] * 100:0.4f}%')
            query_scales = query_scales_dict['query_scales']

        # Candidates for query image
        candidates_result = get_candidates(query_features, self.features, FILTERS['c2'], query_scales)
        candidates = candidates_result['candidates']
        dataset_candidates_count = candidates_result['dataset_candidates_count']

        if candidates.shape[0] == 0:
            if log: print("No similar fragments in database:(")
            return np.array([])
        if log: print(f"\n{candidates.shape[0]} candidates found:")
        for dataset in dataset_candidates_count.keys():
            if log: print(f"\t- {dataset_candidates_count[dataset]} in {dataset}")

        # Distances to query image
        distances_dict = get_distances(database_features=self.features, query_features=query_features,
                                       candidates=candidates, distance_fn=DISTANCES['c2_d_near'], log=log)

        for distance in distances_dict.keys():
            random.shuffle(distances_dict[distance])

        used_tiles = {
            dataset_name: {
                scale: np.array([np.zeros(features[i]['features'].shape) for i in range(features.shape[0])])
                for scale, features in scales.items()
            } for dataset_name, scales in self.features.items()
        }

        results = []
        n = 1
        distances_sorted = sorted(distances_dict.keys())
        for distance in distances_sorted:
            if n > top_n:
                break
            for candidate in distances_dict[distance]:
                dataset_name = candidate["dataset_name"]
                scale = candidate['scale']
                img_idx = candidate["image_idx"]
                x = candidate["x"]
                y = candidate["y"]

                if (used_tiles[dataset_name][scale][img_idx][x:x + width, y:y + height] == 0).all():
                    used_tiles[dataset_name][scale][img_idx][x:x + width, y:y + height] = 1
                    candidate['distance'] = distance
                    candidate['coordinates'] = (x * self.__tile_size, y * self.__tile_size)
                    results.append(candidate)
                    if self.cfg.save_results:
                        img = self.get_image(dataset_name=dataset_name, scale=scale, image_idx=img_idx)
                        results_dir_name = to_absolute_path('Results')
                        if not os.path.exists(results_dir_name):
                            os.makedirs(results_dir_name)
                        Image.fromarray(normalize_image(img[x * self.__tile_size:(x + width) * self.__tile_size,
                                                            y * self.__tile_size:(y + height) * self.__tile_size])) \
                            .save(to_absolute_path(f"{results_dir_name}/top_{n}.png"))
                    n += 1
                    if n > top_n:
                        break
        return np.array(results)
