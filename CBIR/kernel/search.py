import numpy as np
from CBIR.kernel.extractors.base_extractor import BaseFeatureExtractor
from CBIR.kernel.utils import LSH, d_near, d_ave
from CBIR.models import ScaleNet
from tqdm import tqdm
import slideio
import torch
import random


def get_image_features(image: np.ndarray, tile_size: int, extractor: BaseFeatureExtractor, binarizator: LSH = None,
                       log=False):
    width, height = (np.array(image.shape) // tile_size)[:2]
    image_binary_features = np.zeros((width, height))
    with tqdm(total=width * height, disable=not log) as pbar:
        for i in range(width):
            for j in range(height):
                tile = image[i * tile_size:(i + 1) * tile_size,
                       j * tile_size:(j + 1) * tile_size, :]
                image_binary_features[i, j] = binarizator.get_signature(
                    extractor.extract_features_for_tile(tile))
                pbar.update(1)
    return image_binary_features


FILTERS = {
    'c2': lambda q, db: np.any(np.in1d(q, db))
}


def get_candidates(query_features, database_features, filter_fn, query_scales=None):
    width, height = query_features.shape

    candidates = []
    dataset_candidates_count = {}
    for dataset_name, scales in database_features.items():
        for scale, features in scales.items():
            if query_scales is not None and scale not in query_scales:
                continue

            for image_idx in range(features.shape[0]):
                for i in range(features[image_idx]['features'].shape[0] - width + 1):
                    for j in range(features[image_idx]['features'].shape[1] - height + 1):
                        if filter_fn(query_features.ravel(),
                                     features[image_idx]['features'][i:i + width, j:j + height]):
                            if dataset_name not in dataset_candidates_count:
                                dataset_candidates_count[dataset_name] = 0
                            dataset_candidates_count[dataset_name] += 1

                            candidates.append({
                                "dataset_name": dataset_name,
                                "scale": scale,
                                "image_idx": image_idx,
                                "x": i,
                                "y": j,
                            })
    return {
        'candidates': np.array(candidates),
        'dataset_candidates_count': dataset_candidates_count,
    }


DISTANCES = {
    'c2_d_near': lambda q, db: d_near(q, db) + d_near(db, q),
    'c2_d_ave': lambda q, db: d_ave(q, db) + d_ave(db, q),
}


def get_distances(database_features, query_features, candidates, distance_fn, log=True):
    distances_dict = {}
    width, height = query_features.shape

    if log: print(f"Calculating distances:")
    for i in tqdm(range(candidates.shape[0]), disable=not log):
        dataset_name = candidates[i]["dataset_name"]
        scale = candidates[i]["scale"]
        image_idx = candidates[i]["image_idx"]
        x = candidates[i]["x"]
        y = candidates[i]["y"]

        db_features = database_features[dataset_name][scale][image_idx]['features'][x:x + width, y:y + height]
        distance = distance_fn(query_features, db_features)
        if distance not in distances_dict:
            distances_dict[distance] = []
        distances_dict[distance].append(candidates[i])
    return distances_dict


def concat_images(im_1, im_2):
    im_1 = np.rollaxis(im_1, 2, 0)
    im_2 = np.rollaxis(im_2, 2, 0)
    return np.concatenate((im_1, im_2))


def generate_tile(wsi, scale, tile_size):
    scale = wsi.magnification / scale
    if scale < 1:
        return None
    location = (
        random.randint(0, wsi.size[0] - int(tile_size[0] * scale)),
        random.randint(0, wsi.size[1] - int(tile_size[1] * scale))
    )
    try:
        tile = wsi.read_block(rect=(*location,
                                    *((np.array(tile_size) * scale).astype('uint32'))),
                              size=tile_size)
    except:
        return None
    tile = tile.astype('uint8')
    return tile


def detect_scales(scalenet: ScaleNet, query, test_wsi, scales, n_samples=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scalenet.to(device)
    scalenet.eval()

    wsi = slideio.open_slide(test_wsi, 'SVS').get_scene(0)

    scale_probs = []
    for scale in scales:
        samples_batch = np.empty([n_samples, scalenet.in_channels,
                                  scalenet.input_size, scalenet.input_size])
        for i in range(n_samples):
            sample = generate_tile(wsi, scale, (scalenet.input_size, scalenet.input_size))
            while sample is None or sample.mean() / 255 > 0.9:
                sample = generate_tile(wsi, scale, (scalenet.input_size, scalenet.input_size))
            samples_batch[i, ...] = concat_images(query, sample)
        samples_batch /= 255
        samples_batch = torch.FloatTensor(samples_batch).to(device)
        scale_probs.append(torch.mean(scalenet(samples_batch)).item())

    detected_scale_idx = int(np.argmax(scale_probs))
    detected_scales = [scales[detected_scale_idx]]
    if detected_scale_idx > 0: detected_scales.append(scales[detected_scale_idx - 1])
    if detected_scale_idx < len(scales) - 1: detected_scales.append(scales[detected_scale_idx + 1])
    return {'query_scales': sorted(detected_scales),
            'detected_scale': scales[detected_scale_idx],
            'prob': np.max(scale_probs)}
