from CBIR.kernel import DataBase
from hydra.utils import to_absolute_path
from PIL import Image
import numpy as np
from functools import lru_cache


def evaluate(cfg):
    print("=====EVAL STARTED=====")
    database = DataBase(cfg)

    query_images = {}
    for dataset in cfg.classes:
        dataset_name = dataset.name
        query_images[dataset_name] = database.load_images(to_absolute_path(cfg.data_path + dataset_name),
                                                          dataset_name, dataset.n_queries)
        database.extract_features(dataset.name)
    database.serialize(to_absolute_path('CBIR_serialized'))
    # database.deserialize(to_absolute_path('CBIR_serialized'))

    print("=====FEATURES EXTRACTED=====")

    metrics = {dataset.name: [] for dataset in cfg.classes}
    map_norm = 0
    log_search = cfg.log_search
    for k in range(1, cfg.top_n + 1):
        map_norm += 1/k
    for dataset in query_images:
        for image_path in query_images[dataset]:
            normalized_map = 0
            image = np.array(Image.open(image_path))
            search_result = database.search(image, cfg.top_n, log=log_search)
            if len(search_result) > 0:
                for k, candidate in enumerate(search_result, start=1):
                    if candidate['dataset_name'] == dataset:
                        normalized_map += 1 / k
                normalized_map /= map_norm
            else:
                normalized_map = 0
            if log_search: print(f"Normalized MAP@{cfg.top_n} for {image_path} = {normalized_map:0.6f}")
            if log_search: print(search_result)
            metrics[dataset].append(normalized_map)
    print()
    for dataset in metrics.keys():
        print(f"{dataset}: average normalized MAP@{cfg.top_n} = {np.mean(metrics[dataset]):0.6f}")
    print(f"Total average normalized MAP@{cfg.top_n} = "
          f"{np.mean([np.mean(metrics[dataset]) for dataset in metrics.keys()]):0.6f}")
    
