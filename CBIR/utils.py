import numpy as np
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import skew, kurtosis, entropy


def normalize_image(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype('uint8')


class LSH:
    # k - number of bits per signature
    def __init__(self, k_bits: int, n_features: int):
        self.k_bits = k_bits
        self.n_features = n_features
        self.seed = np.random.randn(self.k_bits, n_features)

    # LSH signature generation using random projection
    def get_signature(self, features):
        if self.n_features != len(features):
            raise ValueError("Incorrect shape of features vector")

        res = 0
        for p in self.seed:
            res = res << 1
            if np.dot(p, features) >= 0:
                res |= 1
        return res


def get_features(tile):
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

