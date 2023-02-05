import numpy as np

PRECISION = 1e-16

# Distortion measures
def abs_dist(t: int, u: int) -> float:
    return np.abs(t - u)


def squared_dist(t: int, u: int) -> float:
    return (t - u) ** 2


distortion_measures = {
    "abs_dist": abs_dist,
    "squared_dist": squared_dist,
}

# N.B.: credit for these tools belongs to N. Zaslavsky: https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py

# Information
def xlogx(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def H(p, axis=None):
    """Entropy"""
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """mutual information, I(X;Y)"""
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


# Distributions
def marginal(pXY, axis=1):
    """:return pY (axis = 0) or pX (default, axis = 1)"""
    return pXY.sum(axis)


def conditional(pXY):
    """:return  pY_X"""
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])


def joint(pY_X, pX):
    """:return  pXY"""
    return pY_X * pX[:, None]


def marginalize(pY_X, pX):
    """:return  pY"""
    return pY_X.T @ pX


def bayes(pY_X, pX):
    """:return pX_Y"""
    pXY = joint(pY_X, pX)
    pY = marginalize(pY_X, pX)
    return np.where(pY > PRECISION, pXY.T / pY, 1 / pXY.shape[0])
