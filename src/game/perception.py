"""Functions for computing similarity distributions and utility functions for signaling games."""

import numpy as np
from game.languages import StateSpace, State
from typing import Callable
from analysis.tools import distortion_measures


def generate_dist_matrix(
    universe: StateSpace,
    distortion: str = "squared_dist",
) -> np.ndarray:
    """Given a universe, compute the distortion for every pair of points in the universe.

    Args:
        universe: a StateSpace such that objects bear Euclidean distance relations

        distortion: a string corresponding to the name of a pairwise distortion function on states, one of {'abs_dist', 'squared_dist'}
    """
    return np.array(
        [
            np.array(
                [
                    distortion_measures[distortion](t.weight, u.weight)
                    for u in universe.referents
                ]
            )
            for t in universe.referents
        ]
    )


def generate_sim_matrix(universe: StateSpace, similarity: str, **kwargs) -> np.ndarray:
    """Given a universe, compute a similarity score for every pair of points in the universe.

    NB: this is a wrapper function that generates the similarity matrix using the data contained in each State.

    Args:
        universe: a StateSpace such that objects bear Euclidean distance relations

        similarity: a string corresponding to the name of a pairwise similarity function on states
    """

    sim_func = similarity_functions[similarity]

    return np.array(
        [
            sim_func(
                target=t.weight,
                objects=[u.weight for u in universe.referents],
                **kwargs,
            )
            for t in universe.referents
        ]
    )


##############################################################################
# SIMILARITY / UTILITY functions
##############################################################################
# N.B.: we use **kwargs so that sim_func() can have the same API


def exp(
    target: int,
    objects: np.ndarray,
    gamma: float = 1.0,
    distortion: str = "squared_dist",
    speed: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """The (unnormalied) exponential function sim(x,y) = exp(-gamma * d(x,y)).

    Args:
        target: value of state

        objects: set of points with measurable similarity values

        gamma: perceptual discriminatibility parameter

        distortion: a string corresponding to the name of a pairwise distortion function on states, one of {'abs_dist', 'squared_dist'}

        speed: a positive float to scale utility by, serving as learning rate in learning and speed of evolution in replicator dynamics.

    Returns:
        a similarity matrix representing pairwise inverse distance between states
    """
    exp_term = lambda t, u: -gamma * distortion_measures[distortion](t, u)
    return np.exp(np.array([exp_term(target, u) for u in objects])) * speed


def exp_normed(
    target: int,
    objects: np.ndarray,
    gamma: float = 1.0,
    distortion: str = "squared_dist",
    speed: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """The (normalized) exponential function, aka softmax, sim(x,y) = exp(-gamma * d(x,y)) / Z.

    Args:
        target: value of state

        objects: set of points with measurable similarity values

        gamma: perceptual discriminatibility parameter

        distortion: {`abs_dist`, `squared_dist`} the distance measure to use.

        speed: a positive float to scale utility by, serving as learning rate in learning and speed of evolution in replicator dynamics.


    Returns:
        a similarity matrix representing pairwise inverse distance between states
    """
    exp_arr = exp(target, objects, gamma, distortion)
    return exp_arr / exp_arr.sum() * speed


def nosofsky(
    target: int,
    objects: np.ndarray,
    alpha: float = 0.0,
    speed: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """The (Gaussian) perceptual similarity function given by Nosofsky 1986:

        sim_alpha(target, object) =
        {
            1   if  alpha = 0 and target == object
            0   if  alpha = 0 and target != object

            exp(- (target - object)^2 / alpha^2 )
        }

    where alpha is an imprecision parameter. When alpha = 0, agents perfectly discriminate between states; when alpha -> infty, agents cannot discriminate states at all. (Compare to gamma in exp and sofmax, which is s.t. perfect discrimination at infty, and homogeneity at 0.)


    Args:
        target: value of state

        objects: set of points with measurable similarity values

        alpha: perceptual imprecision parameter

        speed: a positive float to scale utility by, serving as learning rate in learning and speed of evolution in replicator dynamics.
    """
    if alpha < 0:
        raise ValueError(
            f"Imprecision parameter alpha must be nonnegative, received {alpha}."
        )

    if alpha == 0:
        sim_point = lambda u: int(target == u)
    if alpha > 0:
        sim_point = lambda u: np.exp(
            -distortion_measures["squared_dist"](target, u) / (alpha**2)
        )

    return np.array([sim_point(u) for u in objects]) * speed


def nosofsky_normed(
    target: int,
    objects: np.ndarray,
    alpha: float = 0.0,
    speed: float = 1.0,
    **kwargs,    
) -> np.ndarray:
    """The nosofsky similarity function, scaled to [0,1]."""
    sim_mat = nosofsky(target, objects, alpha, speed = 1.0, **kwargs)
    return sim_mat / sim_mat.sum() * speed

similarity_functions = {
    "exp": exp,
    "exp_normed": exp_normed,
    "nosofsky": nosofsky,
    "nosofsky_normed": nosofsky_normed,
}


def sim_utility(x: State, y: State, sim_mat: np.ndarray) -> float:
    return sim_mat[int(x.weight), int(y.weight)]
