"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import (
    run_trials,
    trials_to_df,
    mean_trajectory,
    get_hypothetical_variants,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    kwargs = util.experiment_parameters(config)
    trials = run_trials(**kwargs)

    # generate hypothetical variants
    if kwargs["num_variants"]:
        df_variants = get_hypothetical_variants(trials, kwargs["num_variants"])
        util.save_points_df(fn=config.filepaths.variant_points_save_fn, df=df_variants)

    # collect and save measurements
    if kwargs["trajectory"]:
        # save means
        df_means = mean_trajectory(trials)
        util.save_points_df(fn=config.filepaths.mean_points_save_fn, df=df_means)

    df_points = trials_to_df(trials)
    util.save_points_df(fn=config.filepaths.simulation_points_save_fn, df=df_points)


if __name__ == "__main__":
    main()
