from CBIR.kernel import DataBase
from hydra.utils import to_absolute_path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import slideio


def CBIR_test(cfg):
    database = DataBase(cfg)

    for dataset in cfg.classes:
        dataset_name = dataset.name
        database.load_images(to_absolute_path(cfg.data_path + dataset_name), dataset_name)
        database.extract_features(dataset.name)

    database.serialize(to_absolute_path('CBIR_serialized'))
    database.normalize_features()

    # database.load_svs(path=to_absolute_path('/home/artem/data/PATH-DT-MSU-WSI/WSS1/04.svs'), dataset_name='PATH-DT',
    #                   scales=[5, 10, 15, 20, 25, 30, 35, 40])
    # database.extract_features('PATH-DT')
    # database.serialize(to_absolute_path('CBIR_serialized'))

    # database.deserialize(to_absolute_path(cfg.features_serialization.path))
    # database.normalize_features()

    for query_path in cfg.query:
        print('\n', query_path)
        query = np.array(Image.open(to_absolute_path(query_path)))
        search_result = database.search(query, top_n=10, detect_scale=False, log=True)
        print(search_result)

        plt.rcParams.update({
            'font.size': 20,
        })

        f, axarr = plt.subplots(3, 5)
        f.set_size_inches(28, 18)

        axarr[0, 0].set_title(f'Extractor type: {cfg.feature_extractor_type}')
        for i in range(5):
            axarr[0, i].axis('off')
        axarr[0, 2].imshow(query)
        axarr[0, 2].set_title('Query')

        for i, result in enumerate(search_result):
            cur_ax = axarr[1 + i // 5, i % 5]
            cur_ax.imshow(Image.open(search_result[i]['image_name']))
            cur_ax.set_title(f'Top {i + 1}'
                             f'\nSimilarity_score={10e+5*search_result[i]["distance"]:0.4f}')
            cur_ax.axis('off')
        plt.show()
