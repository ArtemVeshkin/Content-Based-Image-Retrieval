import numpy as np


def normalize_image(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype('uint8')


class LSH:
    # k - number of bits per signature
    def __init__(self, k_bits: int, n_features: int):
        self.k_bits = k_bits
        self.n_features = n_features
        self.seed = np.random.randn(self.k_bits, self.n_features)

    # LSH signature generation using random projection
    def get_signature(self, features):
        if self.n_features != len(features):
            raise ValueError(f"Incorrect shape of features vector: "
                             f"self.n_features={self.n_features}, features={len(features)}")

        res = 0
        for p in self.seed:
            res = res << 1
            if np.dot(p, features) >= 0:
                res |= 1
        return res


class Ident:
    def __init__(self):
        self.k_bits = 0
        self.n_features = 0
        self.seed = 0

    @staticmethod
    def get_signature(features):
        return features


# get number of '1's in binary
def nnz(num):
    if num == 0:
        return 0
    res = 1
    num = np.bitwise_and(num, num - 1)
    while num:
        res += 1
        num = np.bitwise_and(num, num - 1)
    return res


# Average distance in multiple binary code space
def d_ave(q, p):
    q, p = q.ravel().astype('uint32'), p.ravel().astype('uint32')
    if q.shape != p.shape:
        raise ValueError("Shapes of input vectors are different")

    res = 0
    for p_elem in p:
        for q_elem in q:
            res += nnz(np.bitwise_xor(p_elem, q_elem))
        res = res / q.shape[0]
    return res / p.shape[0]


# Nearest distance in multiple binary code space
def d_near(q, p):
    q, p = q.ravel().astype('uint32'), p.ravel().astype('uint32')
    if q.shape != p.shape:
        raise ValueError("Shapes of input vectors are different")

    res = 0
    for p_elem in p:
        res += np.vectorize(lambda x: nnz(np.bitwise_xor(p_elem, x)))(q).min()
    return res / p.shape[0]
