import numpy as np
from altk.effcomm.optimization import Mutation
from game.languages import Signal, SignalingLanguage


class AddSignal(Mutation):
    """Add a random signal to the language."""

    def precondition(self, language: SignalingLanguage, **kwargs) -> bool:
        """Only add a singal if the language size is not at maximum."""
        lang_size = kwargs["lang_size"]
        return len(language) < lang_size

    def mutate(
        self, language: SignalingLanguage, expressions: list[Signal], **kwargs
    ) -> SignalingLanguage:
        """Add a random new signal to the language."""
        new_signal = np.random.choice(expressions)
        while new_signal in language:
            new_signal = np.random.choice(expressions)
        language.add_expression(e=new_signal)
        return language


class RemoveSignal(Mutation):
    """Remove a random signal from the language."""

    def precondition(self, language: SignalingLanguage, **kwargs) -> bool:
        """Only remove a signal if it does not remove the only signal in a language."""
        return len(language) > 1

    def mutate(
        self, language: SignalingLanguage, expressions=None
    ) -> SignalingLanguage:
        """Removes a random signal from the list of expressions of a signaling language.

        Dummy expressions argument to have the same function signature as super().mutate().
        """
        index = np.random.randint(0, len(language) - 1)
        language.pop(index)
        return language


class InterchangeSignal(Mutation):

    """Removes and then adds a random expresion.

    Requires creating AddSignal and RemoveSignal mutations as instance attributes.
    """

    def precondition(self, language: SignalingLanguage, **kwargs) -> bool:
        """Always applies."""
        return True

    def mutate(
        self, language: SignalingLanguage, expressions: list[Signal]
    ) -> SignalingLanguage:
        """Removes and then adds a random expresion."""
        add = AddSignal()
        remove = RemoveSignal()
        return remove.mutate(add.mutate(language, expressions), expressions)
