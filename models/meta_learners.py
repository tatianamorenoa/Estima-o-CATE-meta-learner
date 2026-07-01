from catenets.models.jax import (
    SNet,
    TNet,
    XNet,
    DRNet
)

import numpy as np


def fit_slearner(X, W, Y):

    n_features = X.shape[1]

    model = SNet(
        binary_y=False
    )

    model.fit(X=X, y=Y, w=W)

    cate_hat = np.asarray(model.predict(X))

    return model, cate_hat


def fit_tlearner(X, W, Y):

    n_features = X.shape[1]

    model = TNet(
        binary_y=False
    )

    model.fit(X=X, y=Y, w=W)

    cate_hat = np.asarray(model.predict(X))

    return model, cate_hat


def fit_xlearner(X, W, Y):

    n_features = X.shape[1]

    model = XNet(
        binary_y=False
    )

    model.fit(X=X, y=Y, w=W)

    cate_hat = np.asarray(model.predict(X))

    return model, cate_hat


def fit_drlearner(X, W, Y):

    n_features = X.shape[1]

    model = DRNet(
        binary_y=False
    )

    model.fit(X=X, y=Y, w=W)

    cate_hat = np.asarray(model.predict(X))

    return model, cate_hat