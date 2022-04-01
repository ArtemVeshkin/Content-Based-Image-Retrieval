from CBIR.kernel import DataBase
from hydra.utils import to_absolute_path
from PIL import Image
import numpy as np


def CBIR_test(cfg):
    database = DataBase(cfg)
    # database.load_svs(path=to_absolute_path('/home/artem/data/PATH-DT-MSU-WSI/WSS1/04.svs'), dataset_name='PATH-DT',
    #                   scales=[5, 10, 15, 20, 25, 30, 35, 40])
    # database.extract_features('PATH-DT')
    # database.serialize(to_absolute_path('CBIR_serialized'))

    database.deserialize(to_absolute_path(cfg.features_serialization.path))
    query = np.array(Image.open(to_absolute_path(cfg.query)))
    search_result = database.search(query, top_n=10, detect_scale=True, log=True)
    print(search_result)
