import numpy as np

from . import BaseFeatureExtractor, StatExtractor, ContrastiveFeatureExtractor


class MergedStatContrastiveExtractor(StatExtractor, ContrastiveFeatureExtractor):
    def __init__(self, model_path, model_params):
        ContrastiveFeatureExtractor.__init__(self, model_path, model_params)

    def extract_features_for_tile(self, tile):
        stat_features = StatExtractor.extract_features_for_tile(self, tile)
        contrastive_features = ContrastiveFeatureExtractor.extract_features_for_tile(self, tile)
        merged_features = np.concatenate([stat_features, contrastive_features])
        return merged_features
