from hydra.utils import to_absolute_path
import numpy as np
from CBIR.kernel.extractors import EXTRACTORS
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from skimage.transform import resize


def extractor_visualization(cfg):
    extractor = EXTRACTORS[cfg.feature_extractor.type](*cfg.feature_extractor.args)
    if cfg.visualization_type == 'clusters':
        visualize_clusters(extractor, cfg.clusters)
    elif cfg.visualization_type == 'class_distances':
        visualize_class_distances(extractor, cfg.class_distances)
    else:
        raise ValueError(f"Unknown visualization type {cfg.visualization_type}")


def visualize_clusters(extractor, cfg):
    plt_w, plt_h = cfg.n_plots
    f, axarr = plt.subplots(plt_w, plt_h)
    f.set_figwidth(4 * plt_h)
    f.set_figheight(4 * plt_w)
    for image_dir in cfg.image_dirs:
        features = extract_n_features(extractor, image_dir['path'], cfg.n_tiles)
        for w in range(plt_w):
            for h in range(plt_h):
                axis = random.choices(range(features[0].shape[0]), k=2)
                if axis[0] == axis[1]:
                    axis[1] = features[0].shape[0] - axis[0]
                axis.sort()
                x, y = [], []
                for vec in features:
                    x.append(vec[axis[0]])
                    y.append(vec[axis[1]])
                axarr[w, h].scatter(x, y, label=image_dir['name'], s=12)
                axarr[w, h].set_xlim(-5, 5)
                axarr[w, h].set_ylim(-5, 5)
                axarr[w, h].legend()
    plt.show()


def visualize_class_distances(extractor, cfg):
    candidates, queries = {}, {}
    for image_dir in cfg.image_dirs:
        candidates[image_dir['name']] = extract_n_features(extractor,
                                                           image_dir['path'],
                                                           cfg.n_candidates)
        queries[image_dir['name']] = extract_n_features(extractor,
                                                        image_dir['path'],
                                                        cfg.n_queries)

    n_classes = len(cfg.image_dirs)
    f, axarr = plt.subplots(ncols=n_classes)
    max_val = 0
    class_names = list(candidates.keys())
    for i in range(n_classes):
        distances = []
        for j in range(n_classes):
            dist = 0
            for query in queries[class_names[i]]:
                for candidate in candidates[class_names[j]]:
                    if cfg.metric == 'mse':
                        cur_dist = np.mean((query - candidate) ** 2)
                    elif cfg.metric == 'mae':
                        cur_dist = np.mean(np.abs(candidate - query))
                    else:
                        raise AttributeError(f"Unknown metric: {cfg.metric}")
                    dist += cur_dist
            dist /= cfg.n_candidates * cfg.n_queries
            max_val = max(max_val, dist)
            distances.append(dist)
        axarr[i].bar(class_names, distances)
        axarr[i].set_title(class_names[i])
    for i in range(n_classes):
        axarr[i].set_ylim(0, 1.1 * max_val)
    plt.show()


def get_image_paths(image_dir):
    image_dir = to_absolute_path(image_dir)
    files = os.listdir(image_dir)
    image_names = np.array(
        list(filter(lambda file: file.endswith('.jpg') or
                                 file.endswith('.png') or
                                 file.endswith('.bmp'), files)))
    all_images = np.vectorize(lambda img: f"{image_dir}/{img}")(image_names)
    all_images.sort()
    return all_images


def extract_n_features(extractor, image_dir, n_tiles):
    all_images = get_image_paths(image_dir)

    im_size = extractor.model.input_size
    extracted_features = []
    for _ in range(n_tiles):
        image = random.choice(all_images)
        image = np.array(Image.open(image)) / 255
        image = resize(image, (im_size, im_size))
        features = extractor.extract_features_for_tile(image)
        extracted_features.append(features)
    return extracted_features
