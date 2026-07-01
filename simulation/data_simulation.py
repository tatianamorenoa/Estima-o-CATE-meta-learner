import numpy as np
import pandas as pd


def simular_dataset(
    n=2000,
    v=(0.5, 0.3, 0.2),
    prob=0.5,
    seed=123
    
):
   

    # correlacion
    a, b, c = v
    mean = [0, 0, 0]
    cov = np.array([
        [1.0, a, b],
        [a, 1.0, c],
        [b, c, 1.0]
    ])

    rng = np.random.default_rng(seed=123)
    X = rng.multivariate_normal(mean, cov, size=n)

    X = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])


    erro = np.random.normal(0, 1, size=n )
    
    # prob_w = 1 / (1 + np.exp(-(1 + 0.5*X['x1'] - 0.3*X['x2'] + 0.2*X['x3'])))


    W = np.random.binomial(1, prob, size=n)
    
    # CATE VERDADERO

    cate_real = ( -2.0 + 0.5*X['x1'] + 0.3*X['x2'] - 3.0*X['x3'])

    # RESULTADOS POTENCIALES

    Y0 = (10.0 + 0.8*X['x1'] + 2.3*X['x2'] - 2.0*X['x3'] + erro)

    Y1 = Y0 + cate_real

    # outcome observado
    Y = np.where( W == 1, Y1, Y0)


    data = X.copy()

    data['treatment'] = W
    data['y'] = Y
    data['y0'] = Y0
    data['y1'] = Y1
    data['cate_true'] = cate_real

    return data