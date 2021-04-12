import CBIR
import numpy as np

if __name__ == '__main__':
    database = CBIR.DataBase()
    # database.load_images("C:/Users/Artem/Desktop/Учеба/Обработка Изображений/Курсовая/Prog/data")
    #
    # database.extract_features(64)
    # database.binarization()
    #
    # database.serialize("CBIR.pkl")

    database.deserialize('CBIR.pkl')
    database.search('fragment_1.png', 10)
