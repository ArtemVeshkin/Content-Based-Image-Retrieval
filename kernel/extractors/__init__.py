from .base_extractor import BaseFeatureExtractor
from .stat_extractor import StatExtractor
from .conv_extractor import ConvExtractor

__all__ = ['EXTRACTORS']

EXTRACTORS = {
    'stat': StatExtractor,
    'conv': ConvExtractor,
    'VAE': None
}