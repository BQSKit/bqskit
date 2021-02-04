"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""
from __future__ import annotations

from bqskit.qis.unitary import Unitary
from bqskit.utils.cachedclass import CachedClass


class Gate(Unitary, CachedClass):
    """Gate Base Class."""

    @property
    def name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        return self.__class__.__name__

    @property
    def num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        raise AttributeError(
            'Expected num_params class property for gate %s.'
            % self.name,
        )

    @property
    def radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        raise AttributeError(
            'Expected radixes class property for gate %s.'
            % self.name,
        )

    @property
    def size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        raise AttributeError(
            'Expected size class property for gate %s.'
            % self.name,
        )

    def __repr__(self) -> str:
        return f'{self.name}()'
