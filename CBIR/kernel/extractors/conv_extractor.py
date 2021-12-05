import numpy as np

from . import BaseFeatureExtractor
from tensorflow.keras import models, layers
from hydra.utils import to_absolute_path
from skimage.transform import resize


class ConvExtractor(BaseFeatureExtractor):
    def __init__(self, model_path):
        self.model = models.load_model(to_absolute_path(model_path))
        for i in range(6):
            self.model.pop()
        self.model.add(layers.AveragePooling2D((4, 4)))

    def extract_features_for_tile(self, tile):
        tile = np.expand_dims(resize(tile, self.model.inputs[0].shape[1:]), axis=0)
        return np.array(self.model(tile)[0]).flatten()
