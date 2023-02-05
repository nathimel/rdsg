"""Functions for analyzing communication systems."""

import numpy as np
import pandas as pd
from analysis.rd import compute_rate_distortion
from altk.effcomm.agent import Speaker, Listener
from altk.effcomm.tradeoff import interpolate_data
from scipy.spatial.distance import cdist


def agents_to_point(
    speaker: Speaker,
    listener: Listener,
    prior: np.ndarray,
    dist_mat: np.ndarray,
) -> tuple[float]:
    """Convert the dispositions that parametrized a Speaker (Sender) and Listener (Receiver) to a pair of (rate, distortion) values."""
    return compute_rate_distortion(
        p_x=prior,
        p_xhat_x=agents_to_channel(speaker, listener),
        dist_mat=dist_mat,
    )


def agents_to_channel(
    speaker: Speaker,
    listener: Listener,
    fill_rows=False,
) -> np.ndarray:
    """Compute P(act|state) using
        p(act|state) = sum_signal [p(act|signal)* p(signal|state)]
        ->
        P(act|state) = SR

    Args:
        speaker: A speaker agent, i.e. Sender or LiteralSpeaker

        listener: A listener agent, i.e. Receiver or LiteralListener

    Returns:
        p_cond: the 'communication channel' represented by P(act|state).
    """
    speaker_weights = speaker.normalized_weights()
    listener_weights = listener.normalized_weights()
    p_cond = speaker_weights @ listener_weights
    if fill_rows:
        p_cond = rows_zero_to_uniform(p_cond)
    return p_cond


def rows_zero_to_uniform(mat):
    """Ensure that P(act|state) is a probability distribution, i.e. each row (indexed by a state) is a distribution over acts, sums to exactly 1.0. Necessary when exploring mathematically possible languages which sometimes have that p(signal|state) is a vector of 0s."""

    threshold = 1e-5

    for row in mat:
        # less than 1.0
        if row.sum() and 1.0 - row.sum() > threshold:
            print("row is nonzero and sums to less than 1.0!")
            print(row, row.sum())
            raise Exception
        # greater than 1.0
        if row.sum() and row.sum() - 1.0 > threshold:
            print("row sums to greater than 1.0!")
            print(row, row.sum())
            raise Exception

    return np.array([row if row.sum() else np.ones(len(row)) / len(row) for row in mat])


def interpolate_curve(
    curve_data: pd.DataFrame,
    sampled_data: pd.DataFrame = None,
    max_distortion: float = 0,
) -> pd.DataFrame:
    """Interpolate curve data so that it bounds all explored languages. Use the maximum of distortion values attained by both BA and exploration.

    Args:
        curve_data: a DataFrame of Rate-Distortion curve points from BA algorithm

        sampled_data: an optional DataFrame of the explored languages

        max_distortion: an optional value specifying the maximum distortion
    """
    # interpolate points to max cost of sampled data
    if sampled_data is not None:
        max_cost = max(sampled_data["distortion"].max(), curve_data["distortion"].max())
    else:
        max_cost = curve_data["distortion"].max()

    if max_distortion > max_cost:
        max_cost = max_distortion

    # get curve points as list of pairs
    points = list(curve_data.itertuples(index=False, name=None))

    # altk requres (cost, comp)
    points = np.fliplr(points)
    points = interpolate_data(points=points, min_cost=0, max_cost=max_cost)
    # flip back
    points = np.fliplr(points)

    # as DataFrame
    points_df = pd.DataFrame(data=points, columns=["rate", "distortion"])
    return points_df


def measure_optimality(data: pd.DataFrame, curve_data: pd.DataFrame) -> np.ndarray:
    """Compute the min distance to any point on the frontier, for every point. Requires `data` to contain more than one row."""
    # get curve points as list of pairs
    pareto_points = np.array(
        list(curve_data[["rate", "distortion"]].itertuples(index=False, name=None))
    )
    points = np.array(
        list(data[["rate", "distortion"]].itertuples(index=False, name=None))
    )
    # N.B.: do not interpolate, so you don't measure high-dist random langs as more optimal than they are!

    # Measure closeness of each language to any frontier point
    distances = cdist(points, pareto_points)
    min_distances = np.min(distances, axis=1)

    # max complexity will be achieved by B-A
    max_complexity = pareto_points[:, 0].max()
    # points may have higher cost than pareto max cost
    max_cost = max(
        points[:, 1].max(),
        pareto_points[
            :,
        ].max(),
    )
    # max possible distance is sqrt( max_rate^2 + (max_distortion)^2 )
    max_distance = np.sqrt(max_cost**2 + max_complexity**2)
    min_distances /= max_distance

    optimalities = 1 - min_distances
    return optimalities
