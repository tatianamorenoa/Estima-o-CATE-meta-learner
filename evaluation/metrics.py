import numpy as np


def pehe(cate_true, cate_pred):
    return np.sqrt(
        np.mean(
            (cate_true - cate_pred) ** 2
        )
    )


def bias(cate_true, cate_pred):
    return np.mean(
        cate_pred - cate_true
    )


def rmse(cate_true, cate_pred):
    return np.sqrt(
        np.mean(
            (cate_true - cate_pred) ** 2
        )
    )


def mean_cate_true(cate_true):
    return np.mean(cate_true)


def mean_cate_pred(cate_pred):
    return np.mean(cate_pred)