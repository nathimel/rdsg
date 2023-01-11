"""Script to estimate the Rate Distortion curve as the Pareto frontier."""

import hydra
import os
import numpy as np
import pandas as pd
from misc import util
from analysis.rd import blahut_arimoto
from analysis.measure import interpolate_curve
from simulation.driver import game_parameters, trajectory_points_to_df


def get_curve_points(
    prior: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray = np.linspace(start=0, stop=2**7, num=1500),
    unique=False,
) -> list[tuple[float]]:
    """Convert the Rate Distortion theoretical limit to a list of points."""
    rd = lambda beta: blahut_arimoto(dist_mat, p_x=prior, beta=beta)["final"]
    pareto_points = [rd(beta) for beta in betas]

    # control non-smoothness
    if unique:
        pareto_df = pd.DataFrame(data=pareto_points, columns=["rate", "distortion"])
        pareto_df = pareto_df.drop_duplicates(subset=["rate"])
        pareto_points = list(pareto_df.itertuples(index=False, name=None))

    return pareto_points


def get_counterpart_data(
    ba,
    betas,
    alphas=None,
) -> dict[str, pd.DataFrame]:
    points = []
    trajectories = []
    for i, beta in enumerate(betas):
        result = ba(beta)

        points.append(result["final"])
        df_traj = trajectory_points_to_df(result["trajectory"])

        df_traj["beta"] = beta
        if alphas is not None:
            df_traj["alpha"] = alphas[i]
        trajectories.append(df_traj)
    df_trajectories = pd.concat(trajectories)

    df_points = util.points_to_df(points)
    df_points["beta"] = betas
    if alphas is not None:
        df_points["alpha"] = alphas

    return {
        "points": df_points,
        "trajectories": df_trajectories,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)
    kwargs = util.experiment_parameters(config)
    game_params = game_parameters(**kwargs)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)
    counterparts_fn = os.path.join(game_dir, config.filepaths.counterpart_points_fn)
    counterpart_trajectories_fn = os.path.join(
        game_dir, config.filepaths.counterpart_trajectories_fn
    )

    # B-A gets a bit sparse in low-rate regions for np.linspace
    betas = np.concatenate(
        [
            np.linspace(start=0, stop=0.29, num=33),
            np.linspace(start=0.3, stop=0.9, num=33),
            np.linspace(start=1.0, stop=2**7, num=334),
        ]
    )
    points = get_curve_points(
        game_params["prior"],
        game_params["dist_mat"],
        betas,
        unique=True,
    )
    # curve must be interpolated before notebook analyses
    curve_data = interpolate_curve(
        util.points_to_df(points),
        max_distortion=0,
    )

    util.save_points_df(
        fn=curve_fn,
        df=curve_data,
    )

    ba = lambda beta: blahut_arimoto(
        game_params["dist_mat"],
        game_params["prior"],
        beta,
        ignore_converge=True,  # to get all trajectories same length
        trajectory=True,
    )

    # TODO: use hydra to infer the list of swept alpha values to obtain beta-counterparts, which depends on similarity, distortion
    if "nosofsky" in kwargs["similarity"]:
        # alphas = np.array(range(0, 11, 2)).astype(float)
        alphas = np.array(
            [
                0,
                1,
                2,
                4,
                8,
                16,
            ]
        ).astype(float)
        betas = alphas**-2
        # 0 ** -2 \to \infty, just use 1000
        betas[0] = 1000

        result = get_counterpart_data(ba, betas, alphas)
        df_points = result["points"]
        df_trajectories = result["trajectories"] # N.B. I don't use these in any of my final analyses, but this code is more general/flexible anyway so keeping it.

        # # points
        # df_points = util.points_to_df(ba(betas))
        # df_points["beta"] = betas
        # df_points["alpha"] = alphas

    else:  # exp or exp normed
        betas = [0.1, 0.2, 0.5, 0.6, 0.75, 1, 2, 3, 5, 1000]
        betas = np.array(betas).astype(float)

        result = get_counterpart_data(ba, betas)

        df_points = result["points"]
        df_trajectories = result["trajectories"]

        # df_points = util.points_to_df(ba(betas))
        # df_points["beta"] = betas

    util.save_points_df(counterparts_fn, df_points)
    util.save_points_df(counterpart_trajectories_fn, df_trajectories)


if __name__ == "__main__":
    main()
