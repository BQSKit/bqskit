"""This module implements the Optimizer base class."""
from __future__ import annotations

import abc

from bqskit.ir.circuit import Circuit


class Optimizer(abc.ABC):
    """The Optimizer class."""

    @abc.abstractmethod
    def optimize(self, circuit: Circuit) -> None:
        """Optimize the circuit."""
