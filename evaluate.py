from kernel import DataBase
from hydra.utils import to_absolute_path
import numpy as np


def evaluate(cfg):
    print("=====EVAL STARTED=====")
    database = DataBase(cfg)

    query_images = {}
    for dataset in cfg.classes:
        dataset_name = dataset.name
        query_images[dataset_name] = database.load_images(to_absolute_path(cfg.data_path + dataset_name),
                                                          dataset_name, dataset.n_queries)
        database.extract_binary_features(dataset.name)

    print("=====FEATURES EXTRACTED=====")

    map_metrics = {dataset.name: [] for dataset in cfg.classes}
    map_norm = 0
    for k in range(1, cfg.top_n + 1):
        map_norm += 1/k
    for dataset in query_images:
        for image in query_images[dataset]:
            normalized_map = 0
            search_result = database.search(image, cfg.top_n)
            for k, candidate in enumerate(search_result, start=1):
                if candidate['dataset_name'] == dataset:
                    normalized_map += 1 / k
            normalized_map /= map_norm
            print(f"Normalized MAP@{cfg.top_n} for {image} = {normalized_map:0.6f}")
            map_metrics[dataset].append(normalized_map)
    print()
    for dataset in map_metrics.keys():
        print(f"{dataset}: average normalized MAP@{cfg.top_n} = {np.mean(map_metrics[dataset]):0.6f}")
    print(f"Total average normalized MAP@{cfg.top_n} = "
          f"{np.mean([np.mean(map_metrics[dataset]) for dataset in map_metrics.keys()]):0.6f}")
