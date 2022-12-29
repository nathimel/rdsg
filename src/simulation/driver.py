"""Functions for running one trial of a simulation."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Type, Any

from game.agents import Sender, Receiver
from game.languages import State, Signal, StateSpace, SignalMeaning, SignalingLanguage
from game.signaling_game import SignalingGame
from game import perception

from misc.util import points_to_df

from simulation.dynamics import dynamics_map

def game_parameters(
    num_states: int,
    num_signals: int,
    similarity: str,
    **kwargs,
) -> dict[str, Any]:
    """Construct some default parameters for a signaling game, e.g. the signals, states, agents, utility function, etc.

    Args:
        num_states: number of states in the game

        num_signals: number of signals in the game

        similarity: a string corresponding to the name of similarity function to use for the game (see perception.py)

    Returns:
        a dictionary of parameters to pass directly into the SignalingGame constructor.
    """
    # dummy names for signals, states
    state_names = [i for i in range(num_states)]
    signal_names = [f"signal_{i}" for i in range(num_signals)]

    # Construct the universe of states, and language defined over it
    universe = StateSpace([State(name=str(name), weight=name) for name in state_names])

    # All meanings are dummy placeholders at this stage, but they can be substantive once agents are given a weight matrix.
    dummy_meaning = SignalMeaning(states=[], universe=universe)
    signals = [Signal(form=name, meaning=dummy_meaning) for name in signal_names]

    # Create a seed language to initialize agents.
    seed_language = SignalingLanguage(signals=signals)
    sender = Sender(
        seed_language, name="sender"
    )
    receiver = Receiver(seed_language, name="receiver")

    # specify prior and distortion matrix for all trials
    kwargs["prior_over_states"] = np.ones(num_states) / num_states

    kwargs["dist_mat"] = perception.generate_dist_matrix(universe, kwargs["distortion"])

    # construct utility function
    sim_kwargs = {"distortion": kwargs["distortion"]}
    if similarity == "nosofsky":
        sim_kwargs["alpha"] = kwargs["sim_param"]
    elif similarity in ["exp", "exp_normed"]: 
        sim_kwargs["gamma"] = kwargs["sim_param"]
    utility = perception.generate_sim_matrix(universe, similarity, **sim_kwargs)
    
    # parameters for a signaling game
    return {
        "states": universe.referents,
        "signals": signals,
        "sender": sender,
        "receiver": receiver,
        "utility": utility,
        "prior": kwargs["prior_over_states"],
        "dist_mat": kwargs["dist_mat"],
    }

##############################################################################
# Helper functions for running experiments
##############################################################################

def run_trials(
    *args,
    **kwargs,
) -> list[SignalingGame]:
    """Run a simulation for multiple trials."""
    return [run_simulation(*args, **kwargs) for _ in tqdm(range(kwargs["num_trials"]))]


def run_simulation(
    *args,
    **kwargs,
) -> SignalingGame:
    """Run one trial of a simulation and return the resulting game."""
    game_params = game_parameters(**kwargs)
    dynamics = dynamics_map[kwargs["dynamics"]](SignalingGame(**game_params), **kwargs)
    dynamics.run()
    return dynamics.game


##############################################################################
# Functions for measuring signaling games
##############################################################################

def mean_trajectory(trials: list[SignalingGame]) -> pd.DataFrame:
    """Compute the mean (rate, distortion) trajectory of a game across trials.
    """

    # extrapolate values since replicator dynamic converges early
    lengths = np.array([len(trial.data["points"]) for trial in trials])
    max_length = lengths.max()
    if np.all(lengths == max_length):
        # no need to extrapolate
        points = np.array([np.array(trial.data["points"]) for trial in trials])
    
    else:
        # pad each array with its final value
        extrapolated = []
        for trial in trials:
            points = np.array(trial.data["points"])
            extra = points[-1] * np.ones((max_length, 2)) # 2D array of points
            extra[:len(points)] = points
            extrapolated.append(extra)
        points = np.array(extrapolated)

    mean_traj = np.mean(points, axis=0)
    points = np.squeeze(mean_traj)
    points_df = points_to_df(points)
    points_df["round"] = pd.to_numeric(points_df.index)  # continuous scale
    return points_df

def trial_to_trajectory_df(sg: SignalingGame) -> pd.DataFrame:
    """Get a dataframe of each (rate, distortion) point that obtains after a round, for all rounds of a single SignalingGame."""
    trajectory_points = sg.data["points"]
    points_df = points_to_df(trajectory_points)
    points_df["round"] = pd.to_numeric(points_df.index)  # continuous scale
    return points_df


def trials_to_df(
    signaling_games: list[SignalingGame],
    trajectory: bool = False,
) -> list[tuple[float]]:
    """Compute the pareto points for a list of resulting simulation languages, based on the distributions of their senders.

    Args:
        trials: a list of SignalingGames after convergence

        trajectory: whether for each trial to return a DataFrame of final round points or a DataFrame of all rounds points (i.e., the game trajectory).

    Returns:
        df: a pandas DataFrame of (rate, distortion) points
    """

    if trajectory:
        return pd.concat([trial_to_trajectory_df(sg=sg) for sg in signaling_games])
    
    return points_to_df([sg.data["points"][-1] for sg in signaling_games])


def games_to_languages(games: list[SignalingGame]) -> list[tuple[SignalingLanguage]]:
    """For each game, extract the sender and receiver's language (signal-state mapping)."""
    languages = [(agent.to_language() for agent in [g.sender, g.receiver]) for g in games]
    return languages