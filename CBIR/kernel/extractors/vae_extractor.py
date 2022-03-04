import numpy as np

from . import BaseFeatureExtractor
from hydra.utils import to_absolute_path
import torch
from skimage.transform import resize
from CBIR.models import VAE


class VAEExtractor(BaseFeatureExtractor):
    def __init__(self, model_path, model_params):
        device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)
        self.model = VAE(*model_params)
        self.model.load(to_absolute_path(model_path))
        self.model.move_to_device(self.device)

    def extract_features_for_tile(self, tile):
        input_size = (self.model.input_size, self.model.input_size)
        tile = np.expand_dims(resize(tile, input_size), axis=0)
        tile = np.moveaxis(tile, -1, 1)
        tile = torch.FloatTensor(tile)
        tile = tile.to(self.device)
        mu, log_var = self.model.encode(tile)
        features = self.model.reparameterize(mu, log_var).cpu().detach()
        return np.array(features).flatten()

    def generate_by_features(self, features):
        features = torch.FloatTensor(features).to(self.device)
        generated = np.array(self.model.decode(features).cpu().detach())
        generated = np.moveaxis(generated, 1, 3)
        return generated
