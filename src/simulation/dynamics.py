"""Submodule for simulating simple evolutionary dynamics."""

import copy
import numpy as np
from typing import Any
from tqdm import tqdm

from altk.effcomm.agent import CommunicativeAgent
from analysis.measure import agents_to_point
from game.signaling_game import SignalingGame
from game.perception import sim_utility


class Dynamics:
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        self.game = game

    def run(self):
        raise NotImplementedError


##############################################################################
# Roth-Erev reinforcement learning
##############################################################################


def reward(agent: CommunicativeAgent, policy: dict[str, Any], amount: float) -> None:
    """Reward an agent for a particular referent-expression behavior.

    In a signaling game, the communicative success of Sender and Receiver language protocols evolve under simple reinforcement learning dynamics. The reward function increments an agent's weight matrix at the specified location by the specified amount.

    Args:
        policy: a dict of the form {"referent": referent, "expression": Expression}

        amount: a positive number reprsenting how much to reward the behavior
    """
    if set(policy.keys()) != {"referent", "expression"}:
        raise ValueError(
            f"The argument `policy` must take a dict with keys 'referent' and 'expression'. Received: {policy.keys()}'"
        )
    if amount < 0:
        raise ValueError(f"Amount to reinforce weight must be a positive number.")
    agent.weights[agent.policy_to_indices(policy)] += amount


class RothErevReinforcementLearning(Dynamics):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.num_rounds = kwargs["num_rounds"]

    def run(self):
        """Simulate reinforcement learning of the pair of agents by playing the signaling game and reinforcing actions depending on their success (utiltity)."""

        sender = self.game.sender
        receiver = self.game.receiver
        utility = lambda x, y: sim_utility(x, y, sim_mat=self.game.utility)

        for _ in tqdm(range(self.num_rounds)):

            # track data
            self.game.data["points"].append(
                agents_to_point(
                    speaker=sender,
                    listener=receiver,
                    prior=self.game.prior,
                    dist_mat=self.game.dist_mat,
                )
            )

            # get input to sender
            target = np.random.choice(a=self.game.states, p=self.game.prior)

            # record interaction
            signal = sender.encode(target)
            output = receiver.decode(signal)
            amount = utility(target, output)

            # update agents
            reward(
                agent=sender,
                policy={"referent": target, "expression": signal},
                amount=amount,
            )
            reward(
                agent=receiver,
                policy={"referent": output, "expression": signal},
                amount=amount,
            )

        self.game.sender = sender
        self.game.receiver = receiver


##############################################################################
# Discrete-time replicator dynamic
##############################################################################


class DiscreteTimeReplicatorDynamic(Dynamics):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]

    def run(self):
        """Simulate evolution of strategies in a near-infinite population of agents x using a discrete-time version of the replicator equation:

            x_i' = x_i * ( f_i(x) - sum_j f_j(x_j) )

        Changes in agent type (pure strategies) depend only on their frequency and their fitness.
        """
        U = self.game.utility
        prior = self.game.prior
        n_signals = len(self.game.signals)
        n_states = len(self.game.states)

        normalize_rows = lambda mat: np.nan_to_num(mat / mat.sum(axis=1, keepdims=True))

        sender = self.game.sender
        receiver = self.game.receiver
        S = sender.normalized_weights()
        R = receiver.normalized_weights()

        i = 0
        converged = False
        progress_bar = tqdm(total=self.max_its)
        while not converged:
            i += 1
            progress_bar.update(1)

            S_prev = copy.deepcopy(S)
            R_prev = copy.deepcopy(R)

            # track data
            # use temporary agents and already normed weights
            sender.weights = S
            receiver.weights = R
            self.game.data["points"].append(
                agents_to_point(
                    speaker=sender,
                    listener=receiver,
                    prior=prior,
                    dist_mat=self.game.dist_mat,
                )
            )

            # # re-measure population-sensitive expected utilities for agents
            U_sender = np.array(
                [[R[w] @ U[s] for w in range(n_signals)] for s in range(n_states)]
            )
            S *= U_sender
            S = normalize_rows(S)

            U_receiver = np.array(
                [
                    [(prior * S[:, w]) @ U[s] for w in range(n_signals)]
                    for s in range(n_states)
                ]
            )
            R *= U_receiver
            R = normalize_rows(R)

            # Check for convergence
            if (
                np.abs(S - S_prev).sum() < self.threshold
                and np.abs(R - R_prev).sum() < self.threshold
            ) or (i == self.max_its):
                converged = True

        progress_bar.close()

        # update final sender and receiver
        self.game.sender.weights = S
        self.game.receiver.weights = R


dynamics_map = {
    "reinforcement_learning": RothErevReinforcementLearning,
    "replicator_dynamic": DiscreteTimeReplicatorDynamic,
}
