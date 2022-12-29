import plotnine as pn
import pandas as pd
import numpy as np


def basic_tradeoff_plot(
    pareto_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    sampled_data: pd.DataFrame = None,
) -> pn.ggplot:
    """Get a basic plotnine point plot of languages in a complexity vs comm_cost 2D plot."""
    plot = (
        # Set data and the axes
        pn.ggplot(data=pareto_data, mapping=pn.aes(x="rate", y="distortion"))
        + pn.xlab("Rate I(S;A)")
        + pn.ylab("Distortion D[S, A]")
        + pn.scale_color_cmap("cividis")
    )
    if sampled_data is not None:
        plot = plot + pn.geom_point(  # hypothetical langs bottom layer
            sampled_data,
            color="gray",
            shape="o",
            size=2,
            alpha=0.6,
        )
    plot = plot + pn.geom_point(  # simulation langs
        sim_data,
        color="blue",
        shape="o",
        size=4,
    )
    plot = plot + pn.geom_line(size=2)  # pareto frontier last
    return plot


def time_tradeoff_plot(
    pareto_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    sampled_data: pd.DataFrame,
) -> pn.ggplot:
    """Get a plotnine point plot of languages in a complexity vs comm_cost 2D plot, color coded by rounds to demonstrate an evolutionary trajectory."""

    # final language points for each simulation trial
    sim_final_data = sim_data[sim_data["round"] == sim_data["round"].max()]

    plot = (
        # Set data and the axes
        pn.ggplot(
            data=pareto_data, mapping=pn.aes(x="rate", y="distortion")
        )  # pareto data
        + pn.geom_point(  # sampled langs
            sampled_data,
            color="gray",
            shape="o",
            size=4,
            alpha=0.2,
        )
        + pn.geom_point(  # simulation langs
            data=sim_data,
            mapping=pn.aes(color="round"),
            shape="o",
            alpha=0.2,
            size=4,
        )
        + pn.geom_point(  # final langs
            data=sim_final_data,
            color="orange",
            shape="o",
            size=4,
        )
        + pn.geom_line(size=1)
        + pn.xlab("Rate I(S;A)")
        + pn.ylab("Distortion d(S, A)")
        + pn.scale_color_cmap("cividis")
    )
    return plot
