from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from kernel.utils import normalize_image


def get_nn_features(tile, model):
    tile = resize(tile, model.inputs[0].shape[1:])
    if model is None:
        raise ValueError("Model is not loaded")
    return np.array(model(np.array([tile]))).flatten()


def get_stat_features(tile):
    tile_features = []

    # Calculating features for tile
    # Statistic features
    grayscale_tile = normalize_image(rgb2gray(tile) * 256)
    tile_features.append(np.mean(grayscale_tile))
    tile_features.append(np.std(grayscale_tile))
    tile_features.append(np.std(grayscale_tile) / np.mean(grayscale_tile))
    tile_features.append(skew(grayscale_tile.ravel()))
    tile_features.append(kurtosis(grayscale_tile.ravel()))
    tile_features.append(entropy(grayscale_tile.ravel()))
    # GLCM
    glcm = greycomatrix(grayscale_tile, distances=[5], angles=[0], levels=256, symmetric=True,
                        normed=True)
    tile_features.append(greycoprops(glcm, 'contrast')[0, 0])
    tile_features.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    tile_features.append(greycoprops(glcm, 'homogeneity')[0, 0])
    tile_features.append(greycoprops(glcm, 'ASM')[0, 0])
    tile_features.append(greycoprops(glcm, 'energy')[0, 0])
    tile_features.append(greycoprops(glcm, 'correlation')[0, 0])

    return np.array(tile_features)
