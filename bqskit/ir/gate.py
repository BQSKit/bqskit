"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""
from typing import List

from bqskit.qis.unitary import Unitary


class Gate(Unitary):
    """Gate Base Class."""

    def get_num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if hasattr(self.__class__, 'num_params'):
            return self.__class__.num_params

        raise AttributeError

    def get_radix(self) -> List[int]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr(self.__class__, 'radix'):
            return self.__class__.radix

        raise AttributeError

    def get_gate_size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if hasattr(self.__class__, 'gate_size'):
            return self.__class__.gate_size

        raise AttributeError
