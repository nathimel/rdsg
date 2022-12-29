"""Functions for computing rate distortion."""
import numpy as np
import pandas as pd
from analysis import tools

# Experiment utils
def get_curve_points(
    prior: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray = np.linspace(start=0, stop=2**7, num=1500),
    unique=False,
) -> list[tuple[float]]:
    """Convert the Rate Distortion theoretical limit to a list of points."""
    rd = lambda beta: blahut_arimoto(dist_mat, p_x=prior, beta=beta)
    pareto_points = [rd(beta) for beta in betas]

    # remove non-smoothness
    if unique:
        pareto_df = pd.DataFrame(data=pareto_points, columns=["rate", "distortion"])
        pareto_df = pareto_df.drop_duplicates(subset=["rate"])
        pareto_points = list(pareto_df.itertuples(index=False, name=None))

    return pareto_points


def information_rate(p_x: np.ndarray, p_xhat_x: np.ndarray) -> float:
    """I(X;Xhat)"""
    p_xhatx = tools.joint(pY_X=p_xhat_x, pX=p_x)
    return tools.MI(pXY=p_xhatx)


def total_distortion(
    p_x: np.ndarray, p_xhat_x: np.ndarray, dist_mat: np.ndarray
) -> float:
    """D[X, Xhat] = sum_x p(x) sum_xhat p(xhat|x) * d(x, xhat)"""
    return np.sum(p_x @ (p_xhat_x * dist_mat))


def compute_rate_distortion(
    p_x,
    p_xhat_x,
    dist_mat,
) -> tuple[np.ndarray]:
    """Compute the information rate I(X;Xhat) and total distortion D[X, Xhat] of a joint distribution defind by P(X) and P(Xhat|X).

    Args:
        p_x: (1D array of shape `|X|`) the prior probability of an input symbol (i.e., the source)

        p_xhat_x: (2D array of shape `(|X|, |Xhat|)`) the probability of an output symbol given the input

        dist_mat: (2D array of shape `(|X|, |X_hat|)`) representing the distoriton matrix between the input alphabet and the reconstruction alphabet.

    Returns:
        a tuple containing
        rate: rate (in bits) of compressing X into X_hat
        distortion: total distortion between X, X_hat
    """
    return (
        information_rate(p_x, p_xhat_x),
        total_distortion(p_x, p_xhat_x, dist_mat),
    )


def blahut_arimoto(
    dist_mat: np.ndarray,
    p_x: np.ndarray,
    beta: float,
    max_it: int = 200,
    eps: float = 1e-5,
) -> tuple[float]:
    """Compute the rate-distortion function of an i.i.d distribution

    Args:

        dist_mat: (2D array of shape `(|X|, |X_hat|)`) representing the distoriton matrix between the input alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j]). In this context, X is a random variable representing the state of Nature, and X_hat is a random variable representing actions appropriate.

        p_x: (1D array of shape `|X|`) representing the probability mass function of the source. In this context, the prior over states of nature.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: (int) max number of iterations

        eps: (float) accuracy required by the algorithm: the algorithm stops if there
                is no change in distoriton value of more than 'eps' between consequtive iterations
    Returns:
        a tuple containing
        rate: rate (in bits) of compressing X into X_hat
        distortion: total distortion between X, X_hat
    """
    # start with iid conditional distribution, as p(x) may not be uniform
    p_xhat_x = np.tile(p_x, (dist_mat.shape[1], 1)).T

    # normalize
    p_x /= np.sum(p_x)
    p_xhat_x /= np.sum(p_xhat_x, 1, keepdims=True)

    it = 0
    distortion_prev = 0
    distortion = 2 * eps
    while it < max_it and np.abs(distortion - distortion_prev) > eps:
        it += 1
        distortion_prev = distortion

        # p(x_hat) = sum p(x) p(x_hat | x)
        p_xhat = p_x @ p_xhat_x

        # p(x_hat | x) = p(x_hat) exp(- beta * d(x_hat, x)) / Z
        p_xhat_x = np.exp(-beta * dist_mat) * p_xhat
        p_xhat_x /= np.expand_dims(np.sum(p_xhat_x, 1), 1)

        # update for convergence check
        rate, distortion = compute_rate_distortion(p_x, p_xhat_x, dist_mat)

    return rate, distortion
