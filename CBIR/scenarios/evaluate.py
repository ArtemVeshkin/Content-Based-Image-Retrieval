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

    print(database.features['AT'][0][0]['features'])
    database.normalize_features()
    print(database.features['AT'][0][0]['features'])

    print("=====FEATURES EXTRACTED=====")

    metrics = {dataset.name: [] for dataset in cfg.classes}
    log_search = cfg.log_search
    metric_name = METRIC_TO_METRIC_NAME[cfg.metric]
    top_n = cfg.top_n
    for dataset in query_images:
        for image_path in query_images[dataset]:
            image = np.array(Image.open(image_path))
            search_result = database.search(image, top_n, log=log_search)
            print(dataset, image_path)
            print(search_result)

            metric_value = METRIC_TO_CALC_FN[cfg.metric](search_result, dataset, top_n)
            if log_search: print(f"{metric_name}@{top_n} for {image_path} = {metric_value:0.6f}")
            if log_search: print(search_result)
            metrics[dataset].append(metric_value)

    print()
    for dataset in metrics.keys():
        print(f"{dataset}: average {metric_name}@{top_n} = {np.mean(metrics[dataset]):0.6f}")
    print(f"Total average {metric_name}@{top_n} = "
          f"{np.mean([np.mean(metrics[dataset]) for dataset in metrics.keys()]):0.6f}")


def calc_normalized_map_metric(search_result, target_dataset, top_n):
    @lru_cache(maxsize=None)
    def map_norm():
        map_norm = 0
        for k in range(1, top_n + 1):
            map_norm += 1 / k
        return map_norm

    result = 0
    if len(search_result) > 0:
        for k, candidate in enumerate(search_result, start=1):
            if candidate['dataset_name'] == target_dataset:
                result += 1 / k
        result /= map_norm()

    return result


def calc_mean_metric(search_result, target_dataset, top_n):
    result = 0
    if len(search_result) > 0:
        for k, candidate in enumerate(search_result, start=1):
            if candidate['dataset_name'] == target_dataset:
                result += 1
        result /= top_n

    return result


METRIC_TO_METRIC_NAME = {
    'map': 'Normalized map',
    'mean': 'Mean'
}

# [FN_SIGNATURE] def calc_metric(search_result, target_dataset, top_n)
METRIC_TO_CALC_FN = {
    'map': calc_normalized_map_metric,
    'mean': calc_mean_metric
}