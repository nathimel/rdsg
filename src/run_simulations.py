"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_trials
from game.signaling_game import trials_to_df

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    kwargs = util.experiment_parameters(config)
    trials = run_trials(**kwargs)

    # # collect and save measurements
    df = trials_to_df(trials, kwargs["trajectory"])
    util.save_points_df(fn=config.filepaths.simulation_points_save_fn, df=df)


if __name__ == "__main__":
    main()
