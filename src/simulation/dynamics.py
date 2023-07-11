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


class Learning(Dynamics):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.num_rounds = kwargs["num_rounds"]
        self.round_info = dict()
        self.utility = lambda x, y: sim_utility(x, y, sim_mat=self.game.utility)

    def update(self):
        """Implements the learning mechanism that updates the behavior of agents based on, e.g. the observed round_utility."""
        raise NotImplementedError

    def run(self):
        """Simulate learning for the pair of agents by repeatedly playing the signaling game."""

        for _ in tqdm(range(self.num_rounds)):

            # track data
            # breakpoint()
            self.game.data["points"].append(
                agents_to_point(
                    speaker=self.game.sender,
                    listener=self.game.receiver,
                    prior=self.game.prior,
                    dist_mat=self.game.dist_mat,
                )
            )

            # get input to sender
            target = np.random.choice(a=self.game.states, p=self.game.prior)

            # record interaction
            signal = self.game.sender.encode(target)
            output = self.game.receiver.decode(signal)
            payoff = self.utility(target, output)

            self.round_info = {
                "target": target,
                "signal": signal,
                "output": output,
                "payoff": payoff,
            }

            # update agents
            self.update()


class RothErevReinforcementLearning(Learning):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.learning_rate = kwargs["learning_rate"]

    def update(self):
        """The Roth Erev reinforcement step."""

        target = self.round_info["target"]
        signal = self.round_info["signal"]
        output = self.round_info["output"]
        amount = self.round_info["payoff"] * self.learning_rate

        reward(
            agent=self.game.sender,
            policy={"referent": target, "expression": signal},
            amount=amount,
        )
        reward(
            agent=self.game.receiver,
            policy={"referent": output, "expression": signal},
            amount=amount,
        )


class SpillOverRERL(RothErevReinforcementLearning):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)

    def update(self):
        """The Roth Erev with 'spillover' generalization reinforcement step."""

        target = self.round_info["target"]
        signal = self.round_info["signal"]
        output = self.round_info["output"]
        max_payoff = self.round_info["payoff"] * self.learning_rate

        # TODO: let the spillover / confusion matrix vary independently of utility
        # The 'generalized' reinforcement (all agents' actions)
        for state in self.game.states:

            policy = {"referent": state, "expression": signal}

            # For Sender, reinforce signal (given similar states)
            sender_spillover = self.utility(state, target) * max_payoff
            reward(
                agent=self.game.sender,
                policy=policy,
                amount=sender_spillover,
            )

            # For Receiver, reinforce similar states (given the signal)
            receiver_spillover = self.utility(state, output) * max_payoff
            reward(
                agent=self.game.receiver,
                policy=policy,
                amount=receiver_spillover,
            )


##############################################################################
# Discrete-time replicator dynamic
##############################################################################


class NoisyDiscreteTimeReplicatorDynamic(Dynamics):
    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]

        # TODO: let this confusion matrix vary independently of utility. When I get hydra figured out, I'll remove the extra classes for R.D. and R.L., and just let the noise / generalization be hparams.
        self.confusion = self.game.utility

    def run(self):
        """Simulate evolution of strategies in a near-infinite population of agents x using a discrete-time version of the replicator equation:

            x_i' = x_i * ( f_i(x) - sum_j f_j(x_j) )

        Changes in agent type (pure strategies) depend only on their frequency and their fitness.
        """
        U = self.game.utility
        C = self.confusion
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

            # Update mean Sender
            S *= (R @ U).T # population-dependent utility
            S = normalize_rows(S)
            S = C @ S

            # Update mean Receiver
            R *= prior * (U @ R).T # population-dependent utility
            R = normalize_rows(R)
            R = R @ C

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


class DiscreteTimeReplicatorDynamic(NoisyDiscreteTimeReplicatorDynamic):
    """A special case of the Noisy DTRD when there is no noise, i.e. the confusion matrix is the identity matrix."""

    def __init__(self, game: SignalingGame, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.confusion = np.eye(len(self.game.states))


dynamics_map = {
    "reinforcement_learning": RothErevReinforcementLearning,
    "spillover_learning": SpillOverRERL,
    "replicator_dynamic": DiscreteTimeReplicatorDynamic,
    "noisy_replicator_dynamic": NoisyDiscreteTimeReplicatorDynamic,
}
