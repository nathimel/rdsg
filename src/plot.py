"""Collect all explored points, including those from simulation and the estimated Pareto front and get simple plot."""

import os
import hydra
import pandas as pd
import plotnine as pn
from misc import util, vis
from analysis.measure import interpolate_curve

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # load datapaths
    cwd = os.getcwd()    
    fps = config.filepaths
    curve_fn = os.path.join(cwd, fps.curve_points_save_fn)
    sim_fn = os.path.join(cwd, fps.simulation_points_save_fn)
    plot_fn = os.path.join(cwd, fps.tradeoff_plot_fn)

    # load data
    curve_data = pd.read_csv(curve_fn)
    sim_data = pd.read_csv(sim_fn)

    # interpolate curve to contain explored langs
    curve_data = interpolate_curve(curve_data)

    # get plot
    plot = vis.basic_tradeoff_plot(curve_data, sim_data)
    util.save_plot(plot_fn, plot)

if __name__ == "__main__":
    main()