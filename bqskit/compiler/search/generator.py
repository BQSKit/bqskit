"""This module implements the LayerGenerator base class."""
from __future__ import annotations

import abc
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class LayerGenerator(abc.ABC):
    """
    The LayerGenerator base class.

    Search based synthesis uses the layer generator to generate the root node
    and the successors of a node.
    """

    @abc.abstractmethod
    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """Generate the initial layer for search."""

    @abc.abstractmethod
    def gen_successors(
        self,
        circuit: Circuit,
        data: dict[str, Any],
    ) -> list[Circuit]:
        """Generate the successors of a circuit node."""
