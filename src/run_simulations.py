"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_trials, trials_to_df, mean_trajectory


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    kwargs = util.experiment_parameters(config)
    trials = run_trials(**kwargs)

    # collect and save measurements
    df = trials_to_df(trials, kwargs["trajectory"])
    fn = config.filepaths.simulation_points_save_fn

    if kwargs["trajectory"]:
        # save means
        df = mean_trajectory(trials)
        util.save_points_df(fn=config.filepaths.mean_points_save_fn, df=df)
        # don't overwrite non-trajectory files
        fn = config.filepaths.trajectory_points_save_fn

    util.save_points_df(fn=fn, df=df)


if __name__ == "__main__":
    main()
