"""Script to estimate the Rate Distortion curve as the Pareto frontier."""

import hydra
import os
import numpy as np
from misc import util
from analysis.rd import get_curve_points
from simulation.driver import game_parameters

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)
    kwargs = util.experiment_parameters(config)
    game_params = game_parameters(**kwargs)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)    

    # B-A gets a bit sparse in low-rate regions for np.linspace
    betas = np.concatenate([
        np.linspace(start=0, stop=0.29, num=33),
        np.linspace(start=0.3, stop=0.9, num=33),
        np.linspace(start=1.0, stop=2**7, num=334),
    ])
    points = get_curve_points(
        game_params["prior"],
        game_params["dist_mat"],
        betas,
        unique=True,
    )
    util.save_points_df(
        fn=curve_fn,
        df=util.points_to_df(points),
        )
    
    # TODO: use hydra to infer the list of swept alpha values to obtain beta-counterparts

if __name__ == "__main__":
    main()
