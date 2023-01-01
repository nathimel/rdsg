import os
import yaml
import numpy as np
import pandas as pd
from typing import Any
from plotnine import ggplot
from game.languages import SignalingLanguage

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def set_seed(seed: int) -> None:
    """Sets random seeds."""
    np.random.seed(seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def experiment_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Construct a dict of the main experimental parameters from the strings specified by hydra configs.

    Args:
        configs: a dict-like object resulting from loading an experiment's hydra configs.

    Returns:
        params: a dict containing various data structures for many trials of an experimebnt.
    """

    kwargs = {
        # Game
        "num_states": config.game.size.num_states,
        "num_signals": config.game.size.num_signals,        
        "distortion": config.game.distortion,
        "similarity": config.game.similarity.func,
        "sim_param": config.game.similarity.param,        
        # Explore
        "explore_directions": config.explore_space.directions,
        "seed_gen_size": int(
            float(config.explore_space.pool_size.seed_generation_size)
        ),
        "max_mutations": config.explore_space.pool_size.max_mutations,
        "num_generations": config.explore_space.pool_size.num_generations,
        # Simulation
        "num_trials": config.simulation.num_trials,
        "trajectory": config.simulation.trajectory,
        "dynamics": config.simulation.dynamics.name,
        # Multiprocessing
        "multiprocessing": config.multiprocessing,
        "num_processes": config.num_processes,
    }

    # Dynamics
    if kwargs["dynamics"] == "reinforcement_learning":
        kwargs["num_rounds"] = int(float(config.simulation.dynamics.num_rounds))
        kwargs["learning_rate"] = float(config.simulation.dynamics.learning_rate)

    if kwargs["dynamics"] == "replicator_dynamic":
        kwargs["max_its"] = config.simulation.dynamics.max_its
        kwargs["threshold"] = config.simulation.dynamics.threshold

    return kwargs


def make_path(fn: str) -> None:
    """Creates the path recursively if it does not exist."""
    dirname = os.path.dirname(fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created folder {dirname}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_configs(fn: str) -> dict:
    """Load the configs .yml file as a dict."""
    with open(fn, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)
    print(f"Saved a plot to {fn}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def points_to_df(points: list[tuple[float]]) -> pd.DataFrame:
    """Convert a list of points to a dataframe with rate, distortion as columns."""
    return pd.DataFrame(
        data=points,
        columns=["rate", "distortion"],
    )


def save_points_df(fn: str, df: pd.DataFrame) -> None:
    """Save a dataframe of (Rate, Distortion) points to a CSV."""
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {fn}")


def save_weights_txt(fn, sender, receiver) -> None:
    """Save weights to a txt file."""
    sender_n = sender.weights / sender.weights.sum(axis=1, keepdims=True)
    receiver_n = receiver.weights / receiver.weights.sum(axis=0, keepdims=True)
    np.set_printoptions(precision=2)
    weights_string = f"""Sender
    \n------------------
    \nweights:
    \n{sender.weights}
    \ndistribution:
    \n{sender_n}
    \n

    \nReceiver
    \n------------------
    \nweights:
    \n{receiver.weights}
    \ndistribution:
    \n{receiver_n}
    """
    with open(fn, "w") as f:
        f.write(weights_string)


def save_languages(fn: str, languages: list[SignalingLanguage]) -> None:
    """Save a list of languages to a YAML file."""
    data = {"languages": list(lang.yaml_rep() for lang in languages)}
    with open(fn, "w") as outfile:
        yaml.safe_dump(data, outfile)
