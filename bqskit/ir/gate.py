"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""
from __future__ import annotations

from bqskit.qis.unitary import Unitary
from bqskit.utils.singleton import Singleton


class Gate(Unitary, Singleton):
    """Gate Base Class."""

    @property
    def name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        if hasattr(self.__class__, 'name'):
            return self.__class__.name

        return self.__class__.__name__

    @property
    def num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if hasattr(self.__class__, 'num_params'):
            return self.__class__.num_params

        raise AttributeError(
            'Expected num_params class variable for gate %s.'
            % self.__class__.name,
        )

    @property
    def radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr(self.__class__, 'radixes'):
            return self.__class__.radixes

        raise AttributeError(
            'Expected radixes class variable for gate %s.'
            % self.__class__.name,
        )

    @property
    def size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if hasattr(self.__class__, 'size'):
            return self.__class__.size

        raise AttributeError(
            'Expected size class variable for gate %s.'
            % self.__class__.name,
        )
