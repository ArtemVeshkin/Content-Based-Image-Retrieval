from CBIR.kernel import DataBase
from hydra.utils import to_absolute_path
from PIL import Image
import numpy as np


def CBIR_test(cfg):
    database = DataBase(cfg)
    database.load_images(directory='/home/artem/dev/Content-Based-Image-Retrieval/test_search_data',
                         dataset_name='test')
    database.extract_binary_features('test')
    image = np.array(Image.open('/home/artem/dev/Content-Based-Image-Retrieval/test_search_data/x5_0.jpg'))
    database.search(image, 10)
