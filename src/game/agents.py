"""Senders and Receivers for signaling games."""

import numpy as np
from altk.effcomm.agent import Speaker, Listener
from altk.language.semantics import Meaning
from game.languages import Signal, SignalMeaning, SignalingLanguage
from typing import Any


class Sender(Speaker):
    """A Sender agent in a signaling game chooses a signal given an observed state of nature, according to P(signal | state)."""

    def __init__(
        self,
        language: SignalingLanguage,
        weights=None,
        name: str = None,
    ):
        super().__init__(language, name=name)
        self.shape = (len(self.language.universe), len(self.language))

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(*self.shape)

    def encode(self, state: Meaning) -> Signal:
        """Choose a signal given the state of nature observed, e.g. encode a discrete input as a discrete symbol."""
        index = self.sample_policy(index=self.referent_to_index(state))
        return self.index_to_expression(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(state, signal)` index."""
        return (
            self.referent_to_index(policy["referent"]),
            self.expression_to_index(policy["expression"]),
        )


class Receiver(Listener):
    """A Receiver agent in a signaling game chooses an action=state given a signal they received, according to P(state | signal)."""

    def __init__(self, language: SignalingLanguage, weights=None, name: str = None):
        super().__init__(language, name=name)
        self.shape = (len(self.language), len(self.language.universe))

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(*self.shape)

    def decode(self, signal: Signal) -> SignalMeaning:
        """Choose an action given the signal received, e.g. decode a target state given its discrete encoding."""
        index = self.sample_policy(index=self.expression_to_index(signal))
        return self.index_to_referent(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(signal, state)` index."""
        return (
            self.expression_to_index(policy["expression"]),
            self.referent_to_index(policy["referent"]),
        )


# class BayesianReceiver(Receiver):
#     """A Bayesian reciever chooses an interpretation deterministically given p(s | w)

#     P(S | W) = P(W | S) * P(S) / P(W)
#     """

#     def __init__(self, sender: Sender, prior: np.ndarray, name: str = None):
#         weights = bayes(sender.normalized_weights(), prior)
#         super().__init__(sender.language, weights=weights, name=name)
