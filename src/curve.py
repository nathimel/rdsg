"""Script to estimate the Rate Distortion curve as the Pareto frontier."""

import hydra
import os
import numpy as np
from misc import util
from analysis.rd import get_curve_points
from analysis.measure import interpolate_curve
from simulation.driver import game_parameters


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)
    kwargs = util.experiment_parameters(config)
    game_params = game_parameters(**kwargs)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)
    counterparts_fn = os.path.join(game_dir, config.filepaths.counterpart_points_fn)

    ba = lambda betas: get_curve_points(
        game_params["prior"],
        game_params["dist_mat"],
        betas,
        unique=True,
    )

    # B-A gets a bit sparse in low-rate regions for np.linspace
    betas = np.concatenate(
        [
            np.linspace(start=0, stop=0.29, num=33),
            np.linspace(start=0.3, stop=0.9, num=33),
            np.linspace(start=1.0, stop=2**7, num=334),
        ]
    )
    points = ba(betas)
    # curve must be interpolated before notebook analyses
    curve_data = interpolate_curve(util.points_to_df(points), max_distortion=30)

    util.save_points_df(
        fn=curve_fn,
        df=curve_data,
    )

    # TODO: use hydra to infer the list of swept alpha values to obtain beta-counterparts
    alphas = np.array(range(0, 11, 2)).astype(float)
    betas = alphas**-2
    # 0 ** -2 \to \infty, just use 1000
    betas[0] = 1000
    df = util.points_to_df(ba(betas))
    df["beta"] = betas
    df["alpha"] = alphas
    util.save_points_df(counterparts_fn, df)


if __name__ == "__main__":
    main()
