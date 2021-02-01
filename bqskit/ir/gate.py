"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""
from typing import Iterable

from bqskit.qis.unitary import Unitary


class Gate(Unitary):
    """Gate Base Class."""

    def get_num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if not hasattr(self.__class__, 'num_params'):
            raise AttributeError

        return self.__class__.num_params

    def get_radix(self) -> Iterable[int]:
        """Returns the number of orthogonal states for each qudit."""
        if not hasattr(self.__class__, 'radix'):
            raise AttributeError

        return self.__class__.radix

    def get_gate_size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if not hasattr(self.__class__, 'gate_size'):
            raise AttributeError

        return self.__class__.gate_size
