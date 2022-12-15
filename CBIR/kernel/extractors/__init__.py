from .base_extractor import BaseFeatureExtractor
from .stat_extractor import StatExtractor
from .conv_extractor import ConvExtractor
from .vae_extractor import VAEExtractor
from .ae_extractor import AEExtractor
from .contrastive_extractor import ContrastiveFeatureExtractor
from .merged import MergedStatContrastiveExtractor

__all__ = ['EXTRACTORS']

EXTRACTORS = {
    'stat': StatExtractor,
    'conv': ConvExtractor,
    'AE': AEExtractor,
    'VAE': VAEExtractor,
    'Contrastive': ContrastiveFeatureExtractor,
    'merged_stat_contrastive': MergedStatContrastiveExtractor
}
