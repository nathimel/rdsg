"""Script to estimate the Rate Distortion curve as the Pareto frontier."""

import sys
import pandas as pd
from analysis.rd import get_curve_points
from misc import util
from game.simulation import simulation_parameters


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/curve.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)

    save_fn = configs["filepaths"]["curve_points_save_fn"]

    util.set_seed(configs["random_seed"])
    kwargs = util.experiment_parameters(configs)

    points = get_curve_points(kwargs["prior_over_states"], kwargs["dist_mat"])
    util.save_points_csv(save_fn, points)

    # specify additional points to compare beta to gamma
    gammas = [0.1, 0.5, 0.75, 1.0, 2.0, 5.0, 7.5, 10.0, 100.0, 1000.0]
    gamma_points = get_curve_points(
        kwargs["prior_over_states"], kwargs["dist_mat"], betas=gammas
    )

    gamma_df = pd.DataFrame(data=gamma_points, columns=["rate", "distortion"])
    gamma_df["beta"] = gammas
    fn = save_fn.replace("curve", "gamma")
    gamma_df.to_csv(fn, index=False)
    print(f"Saved {len(gamma_points)} language points to {fn}")


if __name__ == "__main__":
    main()
