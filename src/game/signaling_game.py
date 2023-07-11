"""Class for a signaling game."""

import numpy as np
from game.languages import State, Signal
from game.agents import Sender, Receiver


class SignalingGame:
    """A signaling game is a tuple $(S, M, A, \sigma, \rho, u, P)$ of states, messages, acts, a sender, a receiver, a utility function, and a distribution over states. The sender and receiver have a common payoff, given by a communicative success function.

    In this signaling game, we identify the acts as states. A sim-max game is a signaling game where the utility function gives graded payoff, i.e. is not just the indicator function.
    """

    def __init__(
        self,
        states: list[State],
        signals: list[Signal],
        sender: Sender,
        receiver: Receiver,
        utility: np.ndarray,
        dist_mat: np.ndarray,
        prior: np.ndarray,
    ) -> None:
        """Initialize a signaling game.

        A SignalingGame object can contain data about how its players evolve over time, via e.g. replicator dynamic or learning.

        Args:
            states: the list of states of the environment that function as both input to the sender, and output of the receiver

            signals: the objects produced by the sender, and are input to receiver

            sender: an agent parameterized by a distribution over signals, given states

            receiver: an agent parameterized by a distribution over states, given signals

            utility: a 2D array specifying the pairwise utility of sender inputs and receiver outputs. I.e. the identity matrix if utility is the indicator function, but a (nonlinear) transformation of `dist_mat` in sim_max games.

            dist_mat: a matrix storing the pairwise distance (e.g. 'inverse utility') of sender inputs and receiver outputs.

            prior: an array specifying the probability distribution over states, which can represent the objective frequency of certain states in nature, or the prior belief about them.

        """
        # Main game parameters
        self.states = states
        self.signals = signals
        self.sender = sender
        self.receiver = receiver
        self.utility = utility
        self.dist_mat = dist_mat
        self.prior = prior

        # measurements to track throughout game
        self.data = {
            "accuracy": [],  # communicative success
            "points": [],  # (rate, distortion)
        }
