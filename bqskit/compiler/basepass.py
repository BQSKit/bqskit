"""This module implements the BasePass abstract base class."""
from __future__ import annotations

import abc
from typing import Any

from bqskit.ir.circuit import Circuit


class BasePass(abc.ABC):
    """
    The abstract base for BQSKit compiler passes.

    All BQSKit algorithms must inherit from BasePass to be run within
    the compiler framework. Each child class will need to implement its
    algorithm inside of the :func:`run` method.

    Examples:
        >>> class PrintCNOTCountPass(BasePass):
        ...     def run(self, circ: Circuit, data: dict[str, Any] = {}) -> None:
        ...         print(f"Number of CNOTs: {circ.count(CNOTGate())}")
    """

    @property
    def name(self) -> str:
        """The name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Perform the pass's operation on `circuit`.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to get auxillary information from previous
                passes and to store information for future passes.
                This function should never error based on what is in
                this dictionary.

        Note:
            - This function should be self-contained and have no side effects.
              This is because it will be potentially run in parallel.
        """
