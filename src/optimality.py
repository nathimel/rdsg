"""Script for measuring optimality of semantic systems w.r.t the Pareto frontier = RD curve."""

import os
import hydra
import pandas as pd
from misc import util
from analysis.measure import measure_optimality


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # load datapaths
    cwd = os.getcwd()
    game_dir = cwd.replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)
    fps = config.filepaths
    sim_fn = os.path.join(cwd, fps.simulation_points_save_fn)
    sampled_fn = os.path.join(game_dir, config.filepaths.sampled_points_save_fn)

    # load data
    curve_data = pd.read_csv(curve_fn)
    sim_data = pd.read_csv(sim_fn)
    sampled_data = pd.read_csv(sampled_fn)

    sim_optimality = measure_optimality(sim_data, curve_data)
    sim_data["optimality"] = sim_optimality

    sampled_optimality = measure_optimality(sampled_data, curve_data)
    sampled_data["optimality"] = sampled_optimality

    util.save_points_df(
        fn=sim_fn,
        df=sim_data,
    )
    util.save_points_df(
        fn=sampled_fn,
        df=sampled_data,
    )


if __name__ == "__main__":
    main()
