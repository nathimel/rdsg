"""Functions for running one trial of a simulation."""

import numpy as np
from tqdm import tqdm
from typing import Type, Any

from game.agents import Sender, Receiver
from game.languages import State, Signal, StateSpace, SignalMeaning, SignalingLanguage
from game.signaling_game import SignalingGame
from game import perception

from simulation.dynamics import dynamics_map


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
    """Run one trial of a simulation and return the resulting game.

    Args:

        game_type: the Type of SignalingGame to construct

        num_rounds: the number of rounds to play the simulated game.

    """
    game_params = game_parameters(**kwargs)
    dynamics = dynamics_map[kwargs["dynamics"]](SignalingGame(**game_params), **kwargs)
    dynamics.run()
    return dynamics.game


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
