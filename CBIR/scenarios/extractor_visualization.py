from hydra.utils import to_absolute_path
import numpy as np
from CBIR.kernel.extractors import EXTRACTORS
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from skimage.transform import resize
from tqdm import tqdm


def extractor_visualization(cfg):
    extractor = EXTRACTORS[cfg.feature_extractor.type](*cfg.feature_extractor.args)
    if cfg.visualization_type == 'clusters':
        visualize_clusters(extractor, cfg.clusters)
    elif cfg.visualization_type == 'class_distances':
        visualize_class_distances(extractor, cfg.class_distances)
    elif cfg.visualization_type == 'knn_distance':
        knn_distance(extractor, cfg.knn_distance)
    elif cfg.visualization_type == 'transformation':
        transformation(extractor, cfg.transformation)
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
                    axis[1] = features[0].shape[0] - axis[0] - 1
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
    for image_dir in tqdm(cfg.image_dirs):
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
                    cur_dist = calculate_metric(cfg.metric, candidate, query)
                    dist += cur_dist
            dist /= cfg.n_candidates * cfg.n_queries
            max_val = max(max_val, dist)
            distances.append(dist)
        axarr[i].bar(class_names, distances)
        axarr[i].set_title(class_names[i])
    for i in range(n_classes):
        axarr[i].set_ylim(0, 1.1 * max_val)
    plt.show()


def calculate_metric(metric, candidate, query):
    if metric == 'mse':
        cur_dist = np.mean((query - candidate) ** 2)
    elif metric == 'mae':
        cur_dist = np.mean(np.abs(candidate - query))
    elif metric == 'ce':
        cur_dist = -np.mean(candidate * np.log(query) + (1 - candidate) * np.log(1 - query))
    else:
        raise AttributeError(f"Unknown metric: {metric}")
    return cur_dist


def transformation(extractor, cfg):
    f, axarr = plt.subplots(ncols=(4 + cfg.steps))
    for i in range(cfg.steps + 4):
        axarr[i].axis('off')

    source_path, target_path = '', ''
    for image_dir in cfg.image_dirs:
        if image_dir['name'] == cfg.source:
            source_path = random.choice(get_image_paths(image_dir['path']))
        if image_dir['name'] == cfg.target:
            target_path = random.choice(get_image_paths(image_dir['path']))

    source_image = np.array(Image.open(source_path)) / 255
    target_image = np.array(Image.open(target_path)) / 255
    if cfg.scale != 1.:
        extractor_input = (extractor.model.input_size, extractor.model.input_size)
        size = (np.array(source_image.shape[:2]) * cfg.scale).astype('uint64')
        source_image = resize(source_image[0: size[0], 0: size[1], :], extractor_input)
        target_image = resize(target_image[0: size[0], 0: size[1], :], extractor_input)

    axarr[0].imshow(source_image)
    axarr[cfg.steps + 3].imshow(target_image)

    source_features = extractor.extract_features_for_tile(source_image)
    target_features = extractor.extract_features_for_tile(target_image)

    s_t_vector = (target_features - source_features) / (cfg.steps + 2)
    frames = []
    append_n_frames(frames, source_image, cfg.pre_frames)
    for step in range(cfg.steps + 2):
        cur_state = extractor.generate_by_features(source_features + s_t_vector * step)[0]
        append_n_frames(frames, cur_state, 1)
        axarr[step + 1].imshow(cur_state)
    append_n_frames(frames, target_image, cfg.post_frames)
    frames[0].save(
        to_absolute_path(cfg.gif_path),
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=int(cfg.gif_len / len(frames)),
        loop=0
    )
    plt.show()


def knn_distance(extractor, cfg):
    candidates, queries = {}, {}
    for image_dir in tqdm(cfg.image_dirs):
        candidates[image_dir['name']] = extract_n_features(extractor,
                                                           image_dir['path'],
                                                           cfg.n_candidates)
        queries[image_dir['name']] = extract_n_features(extractor,
                                                        image_dir['path'],
                                                        cfg.n_queries)

    n_classes = len(cfg.image_dirs)
    max_val = 0
    f, axarr = plt.subplots(ncols=n_classes)
    class_names = list(candidates.keys())
    class_names_to_index = {class_names[i]: i for i in range(n_classes)}
    for i in range(n_classes):
        knn_classes = np.zeros(n_classes)
        for query in queries[class_names[i]]:
            distances = []
            for j in range(n_classes):
                for candidate in candidates[class_names[j]]:
                    cur_dist = calculate_metric(cfg.metric, candidate, query)
                    distances.append((cur_dist, class_names[j]))
            distances.sort(key=lambda x: x[0])
            distances = distances[:10]
            for dist in distances:
                knn_classes[class_names_to_index[dist[1]]] += 1
        knn_classes /= len(queries[class_names[i]])
        max_val = max(max_val, np.max(knn_classes))
        colorlist = ['r'] * n_classes
        colorlist[i] = 'g'
        axarr[i].bar(class_names, knn_classes, color=colorlist)
        axarr[i].set_title(class_names[i])
    for i in range(n_classes):
        axarr[i].set_ylim(0, 1.1 * max_val)
    plt.show()


def append_n_frames(frames, frame, n):
    for _ in range(n):
        frames.append(Image.fromarray((frame * 255).astype('uint8')))


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
    for _ in range(min(n_tiles, len(all_images))):
        image = random.choice(all_images)
        image = np.array(Image.open(image)) / 255
        image = resize(image, (im_size, im_size))
        features = extractor.extract_features_for_tile(image)
        extracted_features.append(features)
    return extracted_features
